# Skipgram-notes 代码走读

> 本文是对仓库代码的逐模块拆解（含张量形状、公式对应关系、实现细节）。
> 若只想看一页纸的总览，见 [`OVERVIEW.md`](OVERVIEW.md)。

---

## 一、这个仓库到底做了什么

用**纯 NumPy** 从零实现了 Word2Vec 的 **Skip-gram** 模型，在 **Stanford Sentiment Treebank (SST)** 语料上训练出 10 维词向量，然后做了两件下游验证：

1. **SVD 降维到 2D 可视化**（产出 `CODE/word_vectors0.png`）
2. **余弦相似度 KNN 近义词检索**（终端打印每个目标词的 Top-6 邻居）

没有用 PyTorch/TensorFlow，**梯度全部手推手写**，并用数值梯度（有限差分）做正确性校验。代码骨架来自 CS224n Assignment 2，作者在其上补全了所有 `YOUR CODE HERE` 部分，并加了大量中文推导注释。数学证明在根目录 `Prove_Report.pdf`（该 PDF 有密码保护，本文未引用其内容）。

**当前状态**：已训练完成。`saved_params_40000.npy`（3.0 MB）是 40000 步 SGD 后的词向量矩阵，`saved_state_40000.pickle` 是配套的随机数状态，可直接续训或加载复现结果。

---

## 二、数据管线：`utils/treebank.py`

`StanfordSentiment` 类负责把 SST 原始文本变成可采样的训练流。

| 步骤 | 做法 | 关键数字 |
|------|------|----------|
| 分句分词 | 读 `datasetSentences.txt`，按空格切分并全部小写 | 11855 句 |
| 建词表 | `tokens()` 返回 `word -> idx`，末尾追加 `UNK` | **19539** 词 |
| 高频词下采样 | `rejectProb()`：`p_reject = max(0, 1 - sqrt(t/f))`，`t = 1e-5 × 总词数` | 丢掉 the/a/of 这类词 |
| 语料扩增 | `allSentences()` 把句子列表复制 **30 遍**再逐句随机丢词 | 每遍丢弃结果不同 → 等效数据增强 |
| 窗口采样 | `getRandomContext(C)` 随机取一句一词作中心词，取左右各 ≤C 个词作上下文 | 上下文里会剔除与中心词同形的词 |
| 负采样分布 | `sampleTable()`：词频取 **0.75 次幂**后归一化，展开成长度 10⁶ 的查表数组 | `sampleTokenIdx()` O(1) 采样 |

两个值得注意的点：

- **0.75 次幂**是 Mikolov 原论文的经验值，作用是压平词频分布——让低频词有机会被采为负样本，同时不让高频词垄断。
- `allSentences()` 里的 `sentences * 30` 会在内存里实打实展开 ~35 万个句子，这是首次调用 `run.py` 启动慢的主要原因（之后被缓存到 `self._allsentences`）。

---

## 三、损失函数：`CODE/word2vec.py`

符号约定（与 CS224n handout 一致）：
- `V` = 中心词向量矩阵，`U` = 外围词向量矩阵，**两者都是"一行一个词"**（handout 里是列，代码里转置了，注释中反复强调过这点）
- `v_c` = 中心词向量 `(d,)`，`u_o` = 正样本外围词向量 `(d,)`

### 3.1 Naive Softmax（`naiveSoftmaxLossAndGradient`）

```
y_pred = softmax(U · v_c)              # (N,)  N=词表大小
loss   = -log(y_pred[o])               # 因 y 是 one-hot，CE 退化成这一项
```

梯度（代码里用了一个漂亮的原地技巧）：

```python
y_pred[outsideWordIdx] -= 1            # 就地构造 (ŷ - y)
gradCenterVec   = y_pred @ U           # (N,)·(N,d) -> (d,)   即 Uᵀ(ŷ-y)
gradOutsideVecs = np.outer(y_pred, v_c)  # (N,d)              即 (ŷ-y)v_cᵀ
```

代价：每算一个 (中心词, 上下文词) 对，都要对**全部 19539 个词**做一次 softmax 和一次外积 —— 所以它只用于梯度检验和小规模对照，实际训练不用它。

### 3.2 Negative Sampling（`negSamplingLossAndGradient`，K=10）

```
z_o = σ(u_oᵀ v_c)
z_n = σ(-U[neg]ᵀ v_c)                  # (K,)  一次矩阵乘算完所有负样本
loss = -log(z_o) - Σ log(z_n)
```

梯度：

```python
sum_neg_grad    = (z_n - 1) @ U[neg]                 # (K,)·(K,d) -> (d,)
gradCenterVec   = (z_o - 1) @ u_o - sum_neg_grad
gradOutsideVecs = zeros_like(U)                      # 稀疏！只有 1+K 行非零
gradOutsideVecs[o] = outer(z_o - 1, v_c)
```

**重复负样本的处理**（这是本实现最讲究的一处）。`getNegativeSamples` 只保证负样本 ≠ 正样本，**不保证彼此不重复**；按定义，同一个词被采样 n 次，梯度就要累加 n 次。代码没有写 for 循环逐个累加，而是：

```python
acc = {k: negSampleWordIndices.count(k) * (... 单次梯度 ...)
       for k in set(negSampleWordIndices)}
for i in acc: gradOutsideVecs[i] += acc[i]
```

即**去重后算一次、乘以出现次数**，把 K 次 sigmoid+点积压缩成 `len(set)` 次。文件里保留了朴素的逐个累加版本（"方案2"）作为对照，注释直接写明"逐个更新慢, 因为要重复计算"。

`w2v_my_note.py` 整份文件是这个函数的**演进备份**（全文注释掉的 `slower_negSamplingLossAndGradient`），保留了优化前的写法和中间推导，是笔记而非可执行代码。

### 3.3 Skip-gram 主体（`skipgram`）

对一个窗口内的每个上下文词调用一次损失函数，把结果累加：

```python
for i in outsideWords:
    L, gV, gU = word2vecLossAndGradient(v_c, word2Ind[i], U, dataset)
    loss += L
    gradCenterVecs[center] += gV     # 只有中心词那一行有梯度
    gradOutsideVectors    += gU      # 返回的本身就是全尺寸稀疏矩阵
```

### 3.4 SGD 适配层（`word2vec_sgd_wrapper`）

把"词向量矩阵"伪装成一个普通的 `f(x) -> (loss, grad)`，好让通用的 SGD 直接优化：

- 输入 `wordVectors` 形状 `(2N, d)`：**上半 N 行是 V（center），下半 N 行是 U（outside）**
- 每次调用跑 `batchsize=50` 个随机窗口，窗口大小在 `[1, C]` 随机（等价于对近距离词加权，Mikolov 原做法）
- loss 和 grad 都除以 batchsize 取平均

---

## 四、优化器：`CODE/sgd.py`

朴素 SGD，核心就三行：

```python
loss, mygrad = f(x)
x -= step * mygrad
x = postprocessing(x)
```

外加三个工程细节：

| 机制 | 参数 | 说明 |
|------|------|------|
| 学习率退火 | `ANNEAL_EVERY = 20000`，每次 `step *= 0.5` | 注释写"sgd必须得有, 不然不收敛" |
| Checkpoint | `SAVE_PARAMS_EVERY = 5000` | 存 `.npy` + 随机数状态 `.pickle` |
| 断点续训 | `useSaved=True` | 自动找 `saved_params_*.npy` 里迭代数最大的那个，并把学习率补退火到对应位置 |
| 平滑日志 | `exploss = .95*exploss + .05*loss` | 指数滑动平均，抹掉单 batch 抖动 |

`sanity_check()` 用 `f(x)=x²` 从三个不同初值出发，验证 1000 步后都收敛到 0。

---

## 五、正确性保障：`utils/gradcheck.py`

`gradcheck_naive` 对参数矩阵的**每一个元素**做中心差分：

```
numgrad = [f(x+h) - f(x-h)] / 2h        h = 1e-4
reldiff = |numgrad - grad| / max(1, |numgrad|, |grad|)   要求 < 1e-5
```

关键一招：每次 `f` 调用前都 `random.setstate(rndstate)` **重置随机数状态**。否则负采样每次抽到不同的词，f(x+h) 和 f(x-h) 根本不是同一个函数，数值梯度必然对不上。

`word2vec.py` 的 `test_word2vec()` 用 5 词玩具词表 + 3 维向量，对两种损失分别跑梯度检验，并打印结果与标准答案（`Loss: 11.166…` / `Loss: 16.151…`）对比。

---

## 六、端到端脚本：`CODE/run.py`

```
超参：dimVectors = 10，C = 5，step = 0.3，iterations = 40000，K = 10
```

流程：

1. **初始化**：V 用 `U(-0.5, 0.5)/d` 均匀分布，**U 初始化为全零**，`vstack` 成 `(39078, 10)`
2. **训练**：`sgd(..., useSaved=True)` —— 因为仓库里已有 40000 步的 checkpoint，直接跑会从头加载而不是重新训练
3. **降维可视化**：手写 PCA —— 中心化 → 协方差 `(1/n)·XᵀX` → `np.linalg.svd` → 取前 2 个奇异向量投影。注释里还验证了"协方差是对称阵，所以 U ≈ Vᵀ"（误差 < 1e-15）
4. **KNN**：在 `wordVectors[:N]`（center 那一半）里按余弦相似度找 Top-6

### 可视化结果（`word_vectors0.png`）

24 个目标词的 2D 投影。能看出一些合理聚类：`amazing / wonderful / great` 挤在一起，`snow / woman / cool` 一片，`male` 和 `hail` 被甩到两侧极端（低频词，向量没训练充分）。但整体分离度一般 —— **10 维 + 40k 步 + 1.2 万句语料，本来就到不了很干净的语义空间**。`run.py` 注释里已写明改进方向：维度调到 200（需要 GPU 级算力）。

---

## 七、发现的几处问题

1. **`utils/utils.py` 有模块级副作用代码**（第 39–45 行）：文件底部直接跑了一段 softmax/normalizeRows 的演示并 `print`。由于 `word2vec.py` 会 `from utils.utils import ...`，**每次 import 都会打印一堆随机矩阵**，污染训练日志。这段测试代码应该挪进 `if __name__ == "__main__":`。

2. **`normalizeRows` / `softmax` 用 `try/except` 裸吞异常**来区分 1D 和 2D 输入。可读性和调试友好度都不如 `if x.ndim == 1:`，而且会掩盖真正的错误。

3. **`negSamplingLossAndGradient` 里的 `indices` 变量定义了但没用到**（第 117 行）——是早期版本的遗留。

4. **`acc` 字典里的 `gradOutsideVecs[k]` 项恒为 0**：因为负样本索引保证不等于 `outsideWordIdx`，而此时矩阵除了 `[outsideWordIdx]` 行外全是零。这一项不影响结果，但读起来会让人以为有正样本/负样本索引重合的情况需要处理。

5. **`run.py` 第 84 行的 concatenate 是恒等操作**：`np.concatenate((wv[:N], wv[N:]), axis=0)` 就等于 `wv` 本身。原作业模板里这行的意图应该是 `V + U`（相加合并两套向量，Mikolov 的常见做法），照抄下来变成了空操作。

---

## 八、跑起来

```bash
cd CODE
conda env create -f env.yml && conda activate a2   # python 3.7 + numpy + matplotlib
bash get_datasets.sh                               # 若 utils/datasets/ 为空（当前仓库已自带）

python word2vec.py    # 梯度检验，秒级
python sgd.py         # 优化器 sanity check，秒级
python run.py         # 完整训练 + 可视化 + KNN
```

注意 `run.py` 里 `useSaved=True`，会自动接上 `saved_params_40000.npy`。想从零重训，把该文件移走或改成 `useSaved=False`。
