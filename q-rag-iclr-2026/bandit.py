# bandit.py
import numpy as np
import torch

class UCBBandit:
    """
    UCB1 over discrete arms.
    Each arm corresponds to a 'candidate generator' that builds a mask over chunks.
    """
    def __init__(self, num_arms: int, c: float = 1.0, seed: int = 0):
        self.num_arms = num_arms
        self.c = c
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(num_arms, dtype=np.int64)
        self.values = np.zeros(num_arms, dtype=np.float64)  # mean reward

    def select_arms(self, batch_size: int, eps: float = 0.05):
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            # 还在冷启动：随机试未试过的
            return self.rng.choice(untried, size=batch_size, replace=True).tolist()

        total = self.counts.sum()
        ucb = self.values + self.c * np.sqrt(np.log(total) / self.counts)

        arms = []
        for _ in range(batch_size):
            if self.rng.random() < eps:
                arms.append(int(self.rng.integers(0, self.num_arms)))
            else:
                # 随机打破并列最大值，避免固定偏向某一个 index
                maxv = ucb.max()
                cand = np.where(ucb == maxv)[0]
                arms.append(int(self.rng.choice(cand)))
        return arms

    def update(self, arms, rewards):
        # rewards: list[float], same length as arms
        for a, r in zip(arms, rewards):
            a = int(a)
            self.counts[a] += 1
            n = self.counts[a]
            # incremental mean
            self.values[a] += (float(r) - self.values[a]) / n

    @torch.no_grad()
    def make_mask(self, arms, cur_s_seq, s_par_available_mask: torch.Tensor):
        B, A = s_par_available_mask.shape
        out = torch.zeros((B, A), dtype=torch.bool, device=s_par_available_mask.device)

        for i, arm in enumerate(arms):
            avail = s_par_available_mask[i]
            idx = torch.nonzero(avail, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                continue

            arm = int(arm)
            if arm == 0:
                out[i] = avail

            elif arm == 1:
                # random 20%
                k = max(1, int(0.2 * idx.numel()))
                perm = torch.randperm(idx.numel(), device=idx.device)[:k]
                chosen = idx[perm]
                out[i, chosen] = True

            elif arm == 2:
                # first 256
                k = min(256, idx.numel())
                out[i, idx[:k]] = True

            elif arm == 3:
                # first 1024
                k = min(1024, idx.numel())
                out[i, idx[:k]] = True

            else:
                # fallback
                out[i] = avail

        return out