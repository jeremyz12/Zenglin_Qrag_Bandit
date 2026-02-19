# Q-RAG + Multi-Armed Bandit (UCB) — HotpotQA 实验记录

本仓库基于 `q-rag-iclr-2026` 的训练/评测流程，在 **HotpotQA** 任务上加入了一个 **Multi-Armed Bandit** 模块，用于在检索/上下文构造阶段做**策略选择**，并对比了加入前后的准确度（accuracy）表现。

---

## 1. 我做了什么改动？

### ✅ 新增：`bandit.py`
新增一个轻量的 **Bandit Controller**（多臂策略选择器），实现了：

- **UCB1 (Upper Confidence Bound)**：根据“探索-利用（exploration-exploitation）”原则选择 arm
- **ε-greedy 探索**：用 `epsilon`（默认 0.01）提供少量随机探索，避免早期陷入局部最优
- 多个 **arms（策略候选）**，每个 arm 对应一种“从候选句子/证据中保留哪些” 的规则（通过 `mask` 实现）

---

## 2. Bandit 负责干什么？

在每个 episode / query 过程中，系统面对一个“候选证据池”（比如检索出来的 sentences / passages）时，Bandit 会在若干策略之间做选择：

Bandit 的目标是最大化最终 reward（例如 EM/F1 对应的 episode return），从而在长期训练中偏向更有效的 evidence selection 策略。

---

## 3. Bandit 具体选择什么？（arms 设计）

在 `bandit.py` 里定义了 4 个 arms，每个 arm 通过 `make_mask()` 生成一个 boolean mask 来筛选 evidence：

- **Arm 0: keep_all**  
  保留所有候选（不做筛选）

- **Arm 1: random_20_percent**  
  随机保留约 20% 的候选（强随机下采样）

- **Arm 2: first256**  
  只保留前 **256** 个候选（前缀截断）

- **Arm 3: first1024**  
  只保留前 **1024** 个候选（更长的前缀截断）
  
---

## 4. Bandit 的学习方式：UCB1 + ε-greedy（不是 Q-learning）

你现在的 bandit 是典型的 **context-free bandit**（无状态老虎机）范式：

- **不是 Q-learning / Q-table / TD** 那套（没有 state->action 的 Q(s,a)）
- 只维护每个 arm 的：
  - `counts[a]`: 被选次数  
  - `values[a]`: 平均回报（mean reward）

选择规则（简化描述）：
- 先确保每个 arm 至少被试一次
- 之后用：
  \[
  \text{UCB}(a)=\hat{\mu}_a + \sqrt{\frac{2\ln t}{n_a}}
  \]
  其中 \(\hat{\mu}_a\) 是该 arm 的平均 reward，\(n_a\) 是选择次数，\(t\) 是总步数  
- 以 `epsilon` 小概率随机选 arm 做探索

---

## 5. 准确度提升如何验证？（两次 HotpotQA 对比）

### (A) Q-RAG baseline（Feb18 run）
- **EM = 0.754**
- **F1 = 0.814**
- Mean return = 0.754 ± 0.431

### (B) Q-RAG + Bandit（Feb19 run）
- **EM = 0.796**
- **F1 = 0.847**
- Mean return = 0.796 ± 0.403

### 提升幅度（绝对值）
- **EM: +0.042**（0.754 → 0.796）
- **F1: +0.033**（0.814 → 0.847）

- EM 相对提升约 **+5.57%**
- F1 相对提升约 **+4.05%**

---
