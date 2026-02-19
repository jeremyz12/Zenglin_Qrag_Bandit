from abc import abstractmethod
import numpy as np
from torch import nn, Tensor
import torch
from collections import namedtuple
from typing import Tuple, Dict, List, Any, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import os
from torch.nn.utils.rnn import pad_sequence
from sortedcontainers import SortedList
from functools import reduce
from envs.text_env import TextEnv
from envs.utils import custom_pad_sequence, stack_actions, stack_memory


TrainBatch = namedtuple("TrainBatch", [
   "state", "action", "reward", "next_state", "not_done", "q_values"
])


class ParallelTextEnv:

    def __init__(
        self,
        text_envs: List[TextEnv],
        state_tokenizer: PreTrainedTokenizer,
        action_tokenizer: PreTrainedTokenizer,
        bandit=None
    ):
        self.text_envs = text_envs
        self.state_tokenizer = state_tokenizer
        self.action_tokenizer = action_tokenizer
        self.action_embed_length = text_envs[0].action_embed_length

        self.bandit = bandit

        # record which arm is used for each env's current episode
        # (updated every step; used for update on done)
        self._last_arms = [None for _ in range(len(self.text_envs))]

        self.tmp_data = [[] for _ in range(len(self.text_envs))]
        # self.episodes = []

    def reset(self):
        memory = [e.reset() for e in self.text_envs]
        # reset arm records at new reset
        self._last_arms = [None for _ in range(len(self.text_envs))]
        return memory, stack_memory(memory, self.state_tokenizer, max_length=self.action_embed_length)

    def _apply_bandit_mask(self, s_par, cur_s_seq):
        """
        Apply bandit-based candidate restriction by AND-ing available_mask with bandit_mask.

        Expected bandit interface (minimal):
          - select_arms(batch_size:int) -> List[int]
          - make_mask(arms, cur_s_seq, s_par_available_mask:Tensor) -> Bool Tensor [B, A]
          - update(arms:List[int], rewards:List[float]) -> None   (called on episode done)
        """
        if self.bandit is None:
            return s_par  # no change

        # select an arm for each parallel env
        if not hasattr(self.bandit, "select_arms"):
            return s_par

        B = len(cur_s_seq)
        arms = self.bandit.select_arms(batch_size=B)

        # store chosen arms for later update (per-env)
        for i in range(B):
            self._last_arms[i] = arms[i]

        # build a per-env mask over action candidates
        if not hasattr(self.bandit, "make_mask"):
            return s_par  # cannot mask without make_mask

        bandit_mask = self.bandit.make_mask(
            arms=arms,
            cur_s_seq=cur_s_seq,
            s_par_available_mask=s_par.available_mask,
        )

        if bandit_mask is None:
            return s_par

        # ensure boolean tensor on same device
        if not isinstance(bandit_mask, torch.Tensor):
            bandit_mask = torch.as_tensor(bandit_mask, device=s_par.available_mask.device)
        bandit_mask = bandit_mask.to(device=s_par.available_mask.device, dtype=torch.bool)

        # AND with existing availability
        new_avail = (s_par.available_mask & bandit_mask)

        # IMPORTANT: TextMemory is a namedtuple; stack_memory returns it
        # so we can safely _replace fields.
        return s_par._replace(available_mask=new_avail)

    def rollout(self, n, cur_s_seq, agent, random):

        a_embeds, a_embeds_target = self.get_extra_embeds(agent.critic.action_embed, agent.action_embed_target)
        env_index = list(range(len(self.text_envs)))
        rewards = []

        s_par = stack_memory(cur_s_seq, self.state_tokenizer, max_length=self.action_embed_length)
        new_state_seq = []

        size = 0

        not_dones_seq = [[] for _ in range(len(self.text_envs))]
        q_seq = [[] for _ in range(len(self.text_envs))]
        r_seq = [[] for _ in range(len(self.text_envs))]
        s_seq = [[] for _ in range(len(self.text_envs))]
        a_seq = [[] for _ in range(len(self.text_envs))]
        s_next_seq = [[] for _ in range(len(self.text_envs))]
        r_sum = [0.0 for _ in range(len(self.text_envs))]

        while size < n + len(self.text_envs):

            a_embeds = self.update_embeds(a_embeds, agent.critic.action_embed)
            a_embeds_target = self.update_embeds(a_embeds_target, agent.action_embed_target)

            a_embeds_pos = [emb["rope"] for emb in a_embeds]
            a_embeds_target_pos = [emb["rope"] for emb in a_embeds_target]

            embeds_pt = custom_pad_sequence(
                a_embeds_pos, padding_value=0.0, batch_first=True, pad_to_power_2=False
            )
            embeds_target_pt = custom_pad_sequence(
                a_embeds_target_pos, padding_value=0.0, batch_first=True, pad_to_power_2=False
            )

            # Bandit layer: restrict candidate actions via available_mask
            s_par_masked = self._apply_bandit_mask(s_par, cur_s_seq)

            action, _, q_values = agent.select_action_batch(
                s_par_masked, embeds_pt, embeds_target_pt, random=random
            )
            action = action.cpu().numpy().reshape(-1)
            q_values = q_values.cpu().numpy().reshape(-1)
            new_state_seq = []

            for i, si, ai, qi, env in zip(env_index, cur_s_seq, action, q_values, self.text_envs):
                transition = env.step_and_maybe_reset(
                    ai, self.action_tokenizer, agent.critic.action_embed, agent.action_embed_target
                )
                transition = transition._replace(state=si, q_values=qi)

                new_state_seq.append(transition.new_state)
                size += 1

                s_seq[i].append(transition.state)
                a_seq[i].append(transition.action)
                s_next_seq[i].append(transition.next_state)
                r_seq[i].append(transition.reward)
                not_dones_seq[i].append(1 - int(transition.done))
                q_seq[i].append(qi)
                r_sum[i] += transition.reward

                if transition.done:
                    # keep embeds consistent across reset
                    a_embeds[i], a_embeds_target[i] = transition.embeds
                    rewards.append(r_sum[i])

                    # ✅ Bandit update on episode end (use the last arm for this env)
                    if self.bandit is not None and hasattr(self.bandit, "update"):
                        arm_i = self._last_arms[i]
                        if arm_i is not None:
                            # update expects lists
                            self.bandit.update([arm_i], [float(r_sum[i])])

                    # reset episode reward and clear stored arm for new episode
                    r_sum[i] = 0.0
                    self._last_arms[i] = None

            cur_s_seq = new_state_seq
            s_par = stack_memory(cur_s_seq, self.state_tokenizer, max_length=self.action_embed_length)

        # (the original code sets r_sum = 0.0; keep but it's unused afterwards)
        r_sum = 0.0

        # flatten sequences but drop the last transition per env to align lengths
        s_seq = reduce(lambda e1, e2: e1 + e2, map(lambda e: e[:-1], s_seq))
        a_seq = reduce(lambda e1, e2: e1 + e2, map(lambda e: e[:-1], a_seq))
        s_next_seq = reduce(lambda e1, e2: e1 + e2, map(lambda e: e[:-1], s_next_seq))

        s_stack = stack_memory(s_seq, self.state_tokenizer, max_length=self.action_embed_length)
        next_s_stack = stack_memory(s_next_seq, self.state_tokenizer, max_length=self.action_embed_length)
        a_stack = stack_actions(a_seq, self.action_tokenizer, max_length=self.action_embed_length)

        return new_state_seq, rewards, TrainBatch(
            state=s_stack,
            q_values=torch.FloatTensor(q_seq).to(torch.get_default_device()),
            action=a_stack,
            reward=torch.FloatTensor(r_seq).to(torch.get_default_device()),
            next_state=next_s_stack,
            not_done=torch.IntTensor(not_dones_seq).to(torch.get_default_device()),
        )

    @torch.no_grad()
    def get_extra_embeds(self, embedder: nn.Module, embedder_target: nn.Module) -> np.ndarray:

        embeds = []
        embeds_target = []

        for e in self.text_envs:
            e1, e2 = e.get_extra_embeds(self.action_tokenizer, embedder, embedder_target)
            embeds.append(e1)
            embeds_target.append(e2)

        return list(embeds), list(embeds_target)

    @torch.no_grad()
    def update_embeds(self, embeds, embedder: nn.Module) -> np.ndarray:

        new_embeds = []

        for emb, env in zip(embeds, self.text_envs):
            new_embeds.append(env.update_embeds(emb, embedder))

        return new_embeds
