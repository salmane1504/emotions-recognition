# GoEmotions Multi-Label Training Pipeline — Full Technical Documentation

> **Notebook:** `notebooks/train_model.ipynb`
>
> This document provides an ultra-detailed, mathematically rigorous explanation of every stage implemented in the training pipeline. All formulas are written in LaTeX and render in any viewer that supports KaTeX or MathJax (GitHub, VS Code Markdown preview, Jupyter, etc.).

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Imports and Device Selection](#2-imports-and-device-selection)
3. [Data Loading and Multi-Hot Label Encoding](#3-data-loading-and-multi-hot-label-encoding)
4. [Multi-Label Stratified Splitting](#4-multi-label-stratified-splitting)
5. [Tokenization (Byte-Pair Encoding)](#5-tokenization-byte-pair-encoding)
6. [RoBERTa Architecture](#6-roberta-architecture)
7. [Loss Function — Binary Cross-Entropy with Logits](#7-loss-function--binary-cross-entropy-with-logits)
8. [Optimizer — AdamW](#8-optimizer--adamw)
9. [Learning Rate Scheduler — Linear Warmup with Decay](#9-learning-rate-scheduler--linear-warmup-with-decay)
10. [Gradient Clipping](#10-gradient-clipping)
11. [Sigmoid Activation and Thresholding](#11-sigmoid-activation-and-thresholding)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Training Loop](#13-training-loop)
14. [Model Selection and Checkpointing](#14-model-selection-and-checkpointing)
15. [Final Test Evaluation and Per-Label Report](#15-final-test-evaluation-and-per-label-report)
16. [Hyperparameter Summary](#16-hyperparameter-summary)

---

## 1. Problem Statement

The pipeline solves a **multi-label text classification** task on the [GoEmotions](https://arxiv.org/abs/2005.00547) dataset. Each text sample $x_i$ can be associated with **one or more** emotion labels simultaneously.

Formally, given a corpus of $N$ samples:

$$
\mathcal{D} = \{(x_i, \mathbf{y}_i)\}_{i=1}^{N}
$$

where:

- $x_i$ is a natural-language text string (a Reddit comment).
- $\mathbf{y}_i \in \{0, 1\}^{L}$ is a **multi-hot binary vector** of length $L$ (number of emotion classes, here $L = 28$).
- $y_{i,j} = 1$ means sample $i$ expresses emotion $j$.

The goal is to learn a function $f_\theta: \mathcal{X} \to [0,1]^L$ that maps text to a probability vector, from which binary predictions are obtained by thresholding.

---

## 2. Imports and Device Selection

The notebook uses:

| Library | Purpose |
|---------|---------|
| `torch` | Neural network training, GPU acceleration |
| `transformers` | Pre-trained RoBERTa model and tokenizer |
| `sklearn` | Stratified splitting and evaluation metrics |
| `numpy` / `pandas` | Numerical operations and data handling |
| `tqdm` | Progress bars |

### Device Priority

The notebook selects the compute device in this order:

1. **MPS** (Apple Metal Performance Shaders) — for Apple Silicon GPUs
2. **CUDA** — for NVIDIA GPUs
3. **CPU** — fallback

### Reproducibility

All random number generators are seeded with $\text{SEED} = 42$:

- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)` (if CUDA is available)

This ensures deterministic weight initialization and data shuffling (modulo non-deterministic GPU operations).

---

## 3. Data Loading and Multi-Hot Label Encoding

### Source

The dataset is loaded from `datasets/go_emotions_dataset.csv`. The CSV contains:

| Column type | Examples |
|-------------|----------|
| `text` | The raw Reddit comment |
| Metadata | `id`, `example_very_unclear` |
| Binary emotion columns | `admiration`, `amusement`, `anger`, ..., `surprise` |

### Multi-Hot Construction

Each emotion column $c_j$ already contains $\{0, 1\}$ values. The notebook stacks all $L$ emotion columns into a label matrix:

$$
\mathbf{Y} \in \{0, 1\}^{N \times L}
$$

where row $i$ is the multi-hot vector $\mathbf{y}_i = [y_{i,1}, y_{i,2}, \ldots, y_{i,L}]$.

### Filtering

Only samples with **at least one active label** are retained:

$$
\mathcal{D}_{\text{filtered}} = \left\{ (x_i, \mathbf{y}_i) \;\middle|\; \sum_{j=1}^{L} y_{i,j} > 0 \right\}
$$

This removes unlabeled or ambiguous rows that would contribute no supervised signal.

---

## 4. Multi-Label Stratified Splitting

Standard stratified splitting (as in `sklearn.model_selection.train_test_split`) requires a single categorical label per sample. Multi-label data needs an adapted approach.

### Label-Signature Strategy

**Step 1 — Signature creation.** For each sample $i$, define a signature string by concatenating the names of all active emotions:

$$
\text{sig}(i) = \text{join}\!\Big(\big\{c_j \;\big|\; y_{i,j} = 1\big\}\Big)
$$

For example, a sample labeled with both `joy` and `surprise` gets signature `"joy|surprise"`.

**Step 2 — Rare-signature bucketing.** Signatures that appear fewer than $\tau_{\min}$ times ($\tau_{\min} = 20$) are collapsed into a catch-all bucket `"__OTHER__"`:

$$
\text{key}(i) =
\begin{cases}
\text{sig}(i) & \text{if } \text{count}(\text{sig}(i)) \geq \tau_{\min} \\
\texttt{\_\_OTHER\_\_} & \text{otherwise}
\end{cases}
$$

This prevents `train_test_split` from failing on singleton strata.

**Step 3 — Two-stage split.** Using the bucketed keys as strata:

1. Split full data into **train** (80%) and **temp** (20%) with stratification on $\text{key}(i)$.
2. Split **temp** into **validation** (50% of temp = 10% of total) and **test** (50% of temp = 10% of total) using the same strategy (with $\tau_{\min}/2$ to handle smaller counts).

Final split proportions:

$$
|\mathcal{D}_{\text{train}}| : |\mathcal{D}_{\text{val}}| : |\mathcal{D}_{\text{test}}| \approx 80\% : 10\% : 10\%
$$

---

## 5. Tokenization (Byte-Pair Encoding)

RoBERTa uses **Byte-Pair Encoding (BPE)** tokenization, which operates on byte-level representations of text.

### BPE Algorithm (overview)

1. Start with a vocabulary of individual bytes (256 symbols).
2. Iteratively find the most frequent adjacent pair of tokens in the training corpus.
3. Merge that pair into a single new token.
4. Repeat for a fixed number of merges (RoBERTa uses ~50,000 merges → vocab size ≈ 50,265).

For an input text $x$, the tokenizer produces:

$$
\text{Tokenize}(x) = [t_1, t_2, \ldots, t_T]
$$

where each $t_k \in \{0, 1, \ldots, V-1\}$ is a token ID and $V$ is the vocabulary size.

### Truncation and Padding

- **Max length** $T_{\max} = 128$. Sequences longer than 128 tokens are truncated.
- **Dynamic padding.** Within each mini-batch, shorter sequences are padded to the length of the longest sequence in that batch using the `[PAD]` token. An **attention mask** vector $\mathbf{m} \in \{0,1\}^T$ is created:

$$
m_k =
\begin{cases}
1 & \text{if } t_k \text{ is a real token} \\
0 & \text{if } t_k \text{ is a padding token}
\end{cases}
$$

The attention mask ensures that padding tokens are **ignored** in the self-attention computation.

---

## 6. RoBERTa Architecture

RoBERTa (**R**obustly **o**ptimized **BERT** **a**pproach) is a transformer encoder pre-trained with a masked language modeling (MLM) objective. The notebook uses `roberta-base` (12 layers, 768 hidden dim, 12 attention heads, ~125M parameters).

### Input Representation

Each input token $t_k$ is mapped to an embedding vector:

$$
\mathbf{e}_k = \mathbf{E}_{\text{token}}[t_k] + \mathbf{E}_{\text{pos}}[k]
$$

where:
- $\mathbf{E}_{\text{token}} \in \mathbb{R}^{V \times d}$ is the token embedding matrix ($d = 768$).
- $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{T_{\max} \times d}$ is the positional embedding matrix.

RoBERTa does **not** use token-type embeddings (unlike BERT).

### Transformer Encoder Block

Each of the 12 transformer layers applies:

#### Multi-Head Self-Attention

For $H = 12$ attention heads, each with dimension $d_h = d/H = 64$:

$$
\text{head}_h = \text{Attention}(\mathbf{X}\mathbf{W}_h^Q,\; \mathbf{X}\mathbf{W}_h^K,\; \mathbf{X}\mathbf{W}_h^V)
$$

where the scaled dot-product attention is:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}\right)\mathbf{V}
$$

The scaling factor $\sqrt{d_h}$ prevents the dot products from growing too large, which would push the softmax into regions of extremely small gradients.

The heads are concatenated and projected:

$$
\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\,\mathbf{W}^O
$$

where $\mathbf{W}^O \in \mathbb{R}^{d \times d}$.

#### Feed-Forward Network

Each layer has a position-wise feed-forward network (FFN):

$$
\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\,\mathbf{W}_2 + \mathbf{b}_2
$$

where $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{ff}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d}$, and $d_{ff} = 3072$.

The **GELU** (Gaussian Error Linear Unit) activation is:

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

#### Layer Normalization and Residual Connections

Each sub-layer (attention and FFN) is wrapped with a residual connection and layer normalization:

$$
\mathbf{x}_{\text{out}} = \text{LayerNorm}(\mathbf{x}_{\text{in}} + \text{SubLayer}(\mathbf{x}_{\text{in}}))
$$

Layer normalization across a $d$-dimensional vector:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where $\mu = \frac{1}{d}\sum_{k} x_k$, $\sigma^2 = \frac{1}{d}\sum_{k}(x_k - \mu)^2$, and $\gamma, \beta \in \mathbb{R}^d$ are learnable scale/shift parameters.

### Classification Head

For sequence classification, the `[CLS]` token's hidden state $\mathbf{h}_{\text{CLS}} \in \mathbb{R}^d$ from the final transformer layer is passed through a classification head:

$$
\mathbf{z} = \mathbf{W}_c\,\text{tanh}(\mathbf{W}_p\,\mathbf{h}_{\text{CLS}} + \mathbf{b}_p) + \mathbf{b}_c
$$

where:
- $\mathbf{W}_p \in \mathbb{R}^{d \times d}$, $\mathbf{b}_p \in \mathbb{R}^d$ — pooling layer
- $\mathbf{W}_c \in \mathbb{R}^{d \times L}$, $\mathbf{b}_c \in \mathbb{R}^L$ — final linear projection

The output $\mathbf{z} \in \mathbb{R}^L$ is a vector of **raw logits** (one per emotion label).

---

## 7. Loss Function — Binary Cross-Entropy with Logits

Since this is a **multi-label** problem (labels are not mutually exclusive), the loss is computed **independently** for each label using Binary Cross-Entropy (BCE).

### Per-Sample Loss

For a single sample $(x_i, \mathbf{y}_i)$ with logits $\mathbf{z}_i = f_\theta(x_i) \in \mathbb{R}^L$:

$$
\mathcal{L}_i = -\frac{1}{L}\sum_{j=1}^{L}\Big[y_{i,j}\log\sigma(z_{i,j}) + (1 - y_{i,j})\log(1 - \sigma(z_{i,j}))\Big]
$$

where $\sigma(\cdot)$ is the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Numerically Stable Form (BCEWithLogitsLoss)

PyTorch's `BCEWithLogitsLoss` fuses the sigmoid and log for numerical stability:

$$
\mathcal{L}_i = -\frac{1}{L}\sum_{j=1}^{L}\Big[y_{i,j}\,z_{i,j} - \log(1 + e^{z_{i,j}})\Big]
$$

Using the identity: $\log\sigma(z) = z - \log(1 + e^z)$ and $\log(1 - \sigma(z)) = -\log(1 + e^z)$.

For an even more stable computation, PyTorch uses:

$$
\ell_j = \max(z_{i,j}, 0) - z_{i,j} \cdot y_{i,j} + \log\!\big(1 + e^{-|z_{i,j}|}\big)
$$

### Batch Loss

Over a mini-batch $\mathcal{B}$ of size $B$:

$$
\mathcal{L}_{\text{batch}} = \frac{1}{B}\sum_{i \in \mathcal{B}} \mathcal{L}_i
$$

### Why BCE and not Cross-Entropy?

| | Softmax + Cross-Entropy | Sigmoid + BCE |
|---|---|---|
| **Constraint** | $\sum_j p_j = 1$ (mutual exclusion) | Each $p_j$ is independent |
| **Use case** | Multi-class (one label per sample) | Multi-label (multiple labels per sample) |
| **Activation** | Softmax over all classes | Sigmoid per class |

Since a text can express multiple emotions simultaneously, sigmoid + BCE is the correct choice.

---

## 8. Optimizer — AdamW

The notebook uses **AdamW** (Adam with decoupled weight decay) with learning rate $\eta = 2 \times 10^{-5}$.

### Adam Update Rules

Given parameter $\theta_t$ at step $t$, gradient $g_t = \nabla_\theta \mathcal{L}_t$:

**First moment** (mean of gradients):

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\,g_t
$$

**Second moment** (mean of squared gradients):

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\,g_t^2
$$

**Bias correction** (counteracts zero-initialization of $m_0, v_0$):

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

### AdamW Parameter Update

AdamW decouples weight decay from the adaptive gradient step:

$$
\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\,\theta_t\right)
$$

where:
- $\eta = 2 \times 10^{-5}$ — learning rate
- $\beta_1 = 0.9$ — first moment decay (default)
- $\beta_2 = 0.999$ — second moment decay (default)
- $\epsilon = 10^{-8}$ — numerical stability constant
- $\lambda$ — weight decay coefficient (default $0.01$)

### Why AdamW over Adam?

In standard Adam, weight decay is applied to the gradient before the adaptive scaling, which makes the effective regularization depend on the gradient magnitudes. AdamW fixes this by applying weight decay **directly to the parameters**, yielding more consistent regularization across parameters with different gradient scales.

---

## 9. Learning Rate Scheduler — Linear Warmup with Decay

The notebook uses `get_linear_schedule_with_warmup` from Hugging Face, which implements a **linear warmup** phase followed by a **linear decay** to zero.

### Total Training Steps

$$
S_{\text{total}} = E \times \left\lceil \frac{|\mathcal{D}_{\text{train}}|}{B_{\text{train}}} \right\rceil
$$

where $E = 2$ is the number of epochs and $B_{\text{train}} = 16$ is the training batch size.

### Warmup Steps

$$
S_{\text{warmup}} = \lfloor 0.1 \times S_{\text{total}} \rfloor
$$

### Schedule

At training step $s$, the learning rate multiplier $\alpha(s)$ is:

$$
\alpha(s) =
\begin{cases}
\displaystyle\frac{s}{S_{\text{warmup}}} & \text{if } s < S_{\text{warmup}} \quad \text{(linear warmup)} \\[10pt]
\displaystyle\frac{S_{\text{total}} - s}{S_{\text{total}} - S_{\text{warmup}}} & \text{if } s \geq S_{\text{warmup}} \quad \text{(linear decay)}
\end{cases}
$$

The effective learning rate at step $s$ is:

$$
\eta(s) = \eta_{\text{base}} \cdot \alpha(s)
$$

### Why Warmup?

During early training steps, the model parameters are far from optimal and gradients can be noisy and large. Warming up gradually increases the learning rate to avoid destabilizing large parameter updates before the optimizer's adaptive moment estimates have stabilized.

---

## 10. Gradient Clipping

After computing gradients via backpropagation, the notebook clips gradient norms:

$$
\text{if } \|\nabla_\theta \mathcal{L}\|_2 > \gamma_{\max}, \quad \nabla_\theta \mathcal{L} \leftarrow \gamma_{\max} \cdot \frac{\nabla_\theta \mathcal{L}}{\|\nabla_\theta \mathcal{L}\|_2}
$$

where $\gamma_{\max} = 1.0$ is the maximum allowed $L_2$ norm of the concatenated gradient vector.

The global $L_2$ norm is computed across **all** model parameters:

$$
\|\nabla_\theta \mathcal{L}\|_2 = \sqrt{\sum_{p \in \text{params}} \sum_{k} \left(\frac{\partial \mathcal{L}}{\partial p_k}\right)^2}
$$

This prevents **exploding gradients** — a common issue when fine-tuning large pre-trained models, where a single bad batch can produce abnormally large gradients that catastrophically update the parameters.

---

## 11. Sigmoid Activation and Thresholding

### From Logits to Probabilities

The model outputs raw logits $\mathbf{z} \in \mathbb{R}^L$. Each logit is converted to an independent probability via the sigmoid function:

$$
p_j = \sigma(z_j) = \frac{1}{1 + e^{-z_j}}
$$

Key properties of the sigmoid:
- $\sigma(z) \in (0, 1) \; \forall z \in \mathbb{R}$
- $\sigma(0) = 0.5$
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### Thresholding

Binary predictions are obtained with a fixed threshold $\tau = 0.5$:

$$
\hat{y}_j =
\begin{cases}
1 & \text{if } p_j \geq \tau \\
0 & \text{if } p_j < \tau
\end{cases}
$$

This is equivalent to checking the sign of the logit:

$$
\hat{y}_j = \mathbb{1}[z_j \geq 0]
$$

since $\sigma(z) \geq 0.5 \iff z \geq 0$.

---

## 12. Evaluation Metrics

The notebook computes multiple metrics to capture different aspects of multi-label performance.

### Notation

Let:
- $\hat{Y} \in \{0,1\}^{N \times L}$ — predicted labels
- $Y \in \{0,1\}^{N \times L}$ — ground-truth labels
- $\text{TP}_{j}$ — true positives for label $j$: $\sum_i y_{i,j} \cdot \hat{y}_{i,j}$
- $\text{FP}_{j}$ — false positives for label $j$: $\sum_i (1 - y_{i,j}) \cdot \hat{y}_{i,j}$
- $\text{FN}_{j}$ — false negatives for label $j$: $\sum_i y_{i,j} \cdot (1 - \hat{y}_{i,j})$

### 12.1 Precision, Recall, F1-Score

**Per-label** metrics:

$$
\text{Precision}_j = \frac{\text{TP}_j}{\text{TP}_j + \text{FP}_j}
$$

$$
\text{Recall}_j = \frac{\text{TP}_j}{\text{TP}_j + \text{FN}_j}
$$

$$
F_{1,j} = 2 \cdot \frac{\text{Precision}_j \cdot \text{Recall}_j}{\text{Precision}_j + \text{Recall}_j} = \frac{2\,\text{TP}_j}{2\,\text{TP}_j + \text{FP}_j + \text{FN}_j}
$$

### 12.2 Micro-Averaged Metrics

Micro-averaging **pools all TP/FP/FN across labels** before computing the metric. This gives equal weight to each *prediction* (favoring frequent labels):

$$
\text{Precision}_{\text{micro}} = \frac{\sum_{j=1}^{L} \text{TP}_j}{\sum_{j=1}^{L} (\text{TP}_j + \text{FP}_j)}
$$

$$
\text{Recall}_{\text{micro}} = \frac{\sum_{j=1}^{L} \text{TP}_j}{\sum_{j=1}^{L} (\text{TP}_j + \text{FN}_j)}
$$

$$
F_{1,\text{micro}} = 2 \cdot \frac{\text{Precision}_{\text{micro}} \cdot \text{Recall}_{\text{micro}}}{\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}}
$$

### 12.3 Macro-Averaged Metrics

Macro-averaging **averages the per-label metrics**. This gives equal weight to each *label* (treating rare and common emotions equally):

$$
\text{Precision}_{\text{macro}} = \frac{1}{L}\sum_{j=1}^{L} \text{Precision}_j
$$

$$
\text{Recall}_{\text{macro}} = \frac{1}{L}\sum_{j=1}^{L} \text{Recall}_j
$$

$$
F_{1,\text{macro}} = \frac{1}{L}\sum_{j=1}^{L} F_{1,j}
$$

### 12.4 Subset Accuracy (Exact Match Ratio)

The strictest metric — a sample is correct only if **all** $L$ predictions exactly match:

$$
\text{SubsetAcc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[\hat{\mathbf{y}}_i = \mathbf{y}_i\right]
$$

This is typically low for multi-label problems because a single mispredicted label counts as a full error.

### 12.5 Hamming Loss

Measures the fraction of individual label predictions that are incorrect:

$$
\text{HammingLoss} = \frac{1}{N \cdot L}\sum_{i=1}^{N}\sum_{j=1}^{L} \mathbb{1}\!\left[\hat{y}_{i,j} \neq y_{i,j}\right]
$$

Equivalently, using the XOR operation:

$$
\text{HammingLoss} = \frac{1}{N \cdot L}\sum_{i=1}^{N}\sum_{j=1}^{L} \hat{y}_{i,j} \oplus y_{i,j}
$$

A Hamming loss of 0 means perfect prediction; 1 means every single label is wrong.

---

## 13. Training Loop

The training proceeds for $E = 2$ epochs. Each epoch consists of:

### Forward Pass

For each mini-batch $\mathcal{B}$ of size $B$:

1. Move tensors to device (MPS/CUDA/CPU).
2. Compute logits: $\mathbf{Z} = f_\theta(\mathbf{X})$ where $\mathbf{Z} \in \mathbb{R}^{B \times L}$.
3. Compute BCE loss: $\mathcal{L}_{\text{batch}}$.

### Backward Pass and Parameter Update

1. **Zero gradients:** $\nabla_\theta \leftarrow \mathbf{0}$.
2. **Backpropagation:** Compute $\nabla_\theta \mathcal{L}_{\text{batch}}$ via automatic differentiation (reverse-mode AD through the computational graph).
3. **Gradient clipping:** Cap $\|\nabla_\theta \mathcal{L}\|_2$ at $\gamma_{\max} = 1.0$.
4. **Parameter update:** Apply AdamW step.
5. **Scheduler step:** Update $\eta(s)$.

### Epoch-Level Tracking

After each epoch $e$:

$$
\bar{\mathcal{L}}_{\text{train}}^{(e)} = \frac{1}{|\text{batches}|}\sum_{b=1}^{|\text{batches}|} \mathcal{L}_b
$$

The model is switched to evaluation mode (`model.eval()`) and the full validation set is evaluated **without gradient computation** (`torch.no_grad()`), computing all metrics from Section 12.

---

## 14. Model Selection and Checkpointing

The notebook implements **early stopping by best validation score**:

$$
\theta^* = \arg\max_{\theta^{(e)},\; e = 1, \ldots, E}\; F_{1,\text{micro}}^{\text{val}}\!\left(\theta^{(e)}\right)
$$

After each epoch:
- If $F_{1,\text{micro}}^{\text{val}}$ improves over the best seen so far, save the model's `state_dict` to `artifacts/roberta-goemotions/best_model.pt`.
- The final test evaluation loads this checkpoint.

This prevents overfitting to the training data by selecting the model that generalizes best to unseen validation examples.

---

## 15. Final Test Evaluation and Per-Label Report

### Test Evaluation

The best checkpoint $\theta^*$ is loaded and evaluated on the held-out test set $\mathcal{D}_{\text{test}}$:

$$
\text{Metrics}_{\text{test}} = \text{evaluate}(f_{\theta^*}, \mathcal{D}_{\text{test}})
$$

All metrics from Section 12 are reported.

### Per-Label Report

For each of the $L = 28$ emotion labels, the notebook computes:

$$
\begin{array}{c|cccc}
\text{Label } j & \text{Precision}_j & \text{Recall}_j & F_{1,j} & \text{Support}_j \\
\hline
\text{admiration} & \cdots & \cdots & \cdots & \cdots \\
\text{amusement} & \cdots & \cdots & \cdots & \cdots \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
\end{array}
$$

where $\text{Support}_j = \sum_{i=1}^{N_{\text{test}}} y_{i,j}$ is the number of true positives + false negatives (i.e., total ground-truth positive examples for label $j$ in the test set).

The table is sorted by $F_{1,j}$ in descending order to quickly identify the best- and worst-performing emotions.

---

## 16. Hyperparameter Summary

| Hyperparameter | Symbol | Value |
|---|---|---|
| Seed | $s$ | 42 |
| Max sequence length | $T_{\max}$ | 128 |
| Train batch size | $B_{\text{train}}$ | 16 |
| Eval batch size | $B_{\text{eval}}$ | 32 |
| Learning rate | $\eta$ | $2 \times 10^{-5}$ |
| Epochs | $E$ | 2 |
| Warmup ratio | — | 10% of total steps |
| Gradient clip norm | $\gamma_{\max}$ | 1.0 |
| Decision threshold | $\tau$ | 0.5 |
| Weight decay | $\lambda$ | 0.01 (AdamW default) |
| Adam $\beta_1$ | $\beta_1$ | 0.9 |
| Adam $\beta_2$ | $\beta_2$ | 0.999 |
| Adam $\epsilon$ | $\epsilon$ | $10^{-8}$ |
| Number of labels | $L$ | 28 |
| Model hidden dim | $d$ | 768 |
| Attention heads | $H$ | 12 |
| Transformer layers | — | 12 |
| FFN inner dim | $d_{ff}$ | 3072 |
| Vocabulary size | $V$ | 50,265 |

---

## Appendix: End-to-End Data Flow Diagram

```
┌─────────────┐
│  Raw CSV     │
│  (N × cols)  │
└──────┬──────┘
       │ filter rows with ≥1 label
       ▼
┌─────────────────────┐
│ Multi-hot labels Y  │
│   (N' × L)          │
└──────┬──────────────┘
       │ signature-based stratified split
       ▼
┌──────────────────────────────────────┐
│ Train (80%)  │  Val (10%)  │  Test (10%) │
└──────┬───────┴─────┬───────┴──────┬──────┘
       │ BPE tokenize (max_len=128) │
       ▼             ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│DataLoader│  │DataLoader│  │DataLoader│
│ B=16     │  │ B=32     │  │ B=32     │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     ▼              │              │
┌────────────────┐  │              │
│ RoBERTa + Head │  │              │
│  → logits z    │  │              │
│  → BCE loss    │  │              │
│  → AdamW step  │──┘              │
│  → scheduler   │                 │
│  → grad clip   │                 │
└────┬───────────┘                 │
     │ best ckpt (micro-F1)       │
     ▼                             │
┌────────────────┐                 │
│ Load best θ*   │─────────────────┘
│ Evaluate test  │
│ Per-label F1   │
└────────────────┘
```
