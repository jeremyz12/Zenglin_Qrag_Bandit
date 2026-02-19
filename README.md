# Q-RAG with Multi-Armed Bandit (UCB) on HotpotQA

本仓库基于 `q-rag-iclr-2026` 的训练与评测流程，在 **HotpotQA** 任务中引入一个轻量的 **Multi-Armed Bandit（MAB）** 组件，用于在检索后证据筛选/上下文构造阶段进行策略选择，并报告加入该组件前后的准确度指标变化。

---

## 1. 实验设置概述（Experimental Setup）

- **Task / Dataset**: HotpotQA
- **Evaluation script**: `eval_retriever.py`
- **Episode budget**: 7405（`num_samples=-1`）
- **Max retrieves**: 3（`+envs.max_steps=3`）
- **Seed**: 42
- **Checkpoints**:
  - Baseline: `runs/Feb18_23-19-21_PQN_hotpotqa/model_best.pt`
  - +Bandit: `runs/Feb19_00-32-46_PQN_hotpotqa/model_best.pt`

---

## 2. 方法：Bandit 控制的证据筛选（Bandit-Controlled Evidence Selection）

### 2.1 组件功能定位
新增的 Bandit 组件用于在每个 query/episode 的“候选证据集合”上，选择一种 **evidence selection / truncation 策略（arm）** 来构造最终输入上下文。其优化目标是提升下游任务回报（reward），从而间接提升 **Exact Match（EM）** 与 **F1** 等指标。

### 2.2 Arms（策略集合）
当前实现包含 4 个候选策略（arms），通过 `make_mask()` 生成 boolean mask 来筛选证据条目：

- **Arm 0 — keep_all**：保留全部候选证据
- **Arm 1 — random_20_percent**：随机保留约 20% 候选证据
- **Arm 2 — first256**：仅保留前 256 条候选证据（prefix truncation）
- **Arm 3 — first1024**：仅保留前 1024 条候选证据（prefix truncation）

### 2.3 选择与更新规则（UCB1 + ε-greedy）

Bandit 使用 **UCB1 (Upper Confidence Bound)** 作为主选择规则，并结合 **ε-greedy** 进行小概率随机探索。

- 初始化：保证每个 arm 至少被选择一次，避免 `n_a = 0`。
- UCB1：选择使 `UCB(a)` 最大的 arm，其中  
  `UCB(a) = μ̂_a + √( (2 ln t) / n_a )`

其中：
- `μ̂_a`：arm `a` 的历史平均回报
- `n_a`：arm `a` 被选择次数
- `t`：总选择次数（total pulls）

- ε-greedy：以概率 `epsilon` 随机选 arm；否则（`1 - epsilon`）按 UCB 最大值选择。

---

## 3. 代码改动摘要（Code Changes）
### 3.1 新增文件
  bandit.py
    实现一个轻量的 Bandit controller，包括：
   - UCB1 分数计算
   - ε-greedy 探索
   - arm 统计量维护（均值/计数）
   - 基于 mask 的证据筛选策略接口

### 3.2 集成点（Integration Points）

在检索后 / 上下文构造阶段接入 Bandit：
  选择 arm → 生成 mask → 筛选候选证据 → 构造最终上下文
  episode 结束后使用回报更新所选 arm 的统计量（incremental update）

---

## 4. 结果：HotpotQA 准确度对比（Results）

使用 eval_retriever.py 在相同评测设置下进行对比：

### 4.1 Baseline（Q-RAG）
 - EM = 0.754
 - F1 = 0.814
 - Mean return = 0.754 ± 0.431 (std)

### 4.2 Q-RAG + Bandit（UCB）
 - EM = 0.796
 - F1 = 0.847
 - Mean return = 0.796 ± 0.403 (std)

### 4.3 增益（Absolute / Relative Gain）

 - EM: +0.042（0.754 → 0.796），相对提升 +5.57%
 - F1: +0.033（0.814 → 0.847），相对提升 +4.05%

---

## 5. 复现实验（Reproducibility）

示例评测命令（PowerShell）：

python .\eval_retriever.py `
  pretrained_path="runs/Feb18_23-19-21_PQN_hotpotqa" `
  num_samples=-1 `
  seed=42 `
  +envs.max_steps=3

python .\eval_retriever.py `
  pretrained_path="runs/Feb19_00-32-46_PQN_hotpotqa" `
  num_samples=-1 `
  seed=42 `
  +envs.max_steps=3

---

## 6. 后续工作（Planned Work）

  跨数据集评测（Generalization）
  将同一 Bandit 机制扩展到其他数据集（例如 README 中提到的 Babilong / Musique 等），在一致的评测协议下报告指标变化。
  
  单一策略消融（Ablation: Arm 3 only）
  进行仅启用 Arm 3（first1024） 的对照实验，以区分以下两类贡献来源：
    “Bandit 自适应选择”带来的收益
    “固定 first1024 截断策略”本身带来的收益
