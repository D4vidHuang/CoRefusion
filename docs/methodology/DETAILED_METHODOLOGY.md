# Detailed Methodology Guide

This document provides a detailed explanation of the CoRefusion methodology for researchers who want to understand or reproduce the approach.

## Table of Contents

1. [Overview](#overview)
2. [Problem Formulation](#problem-formulation)
3. [Model Architecture](#model-architecture)
4. [Training Procedure](#training-procedure)
5. [Inference Process](#inference-process)
6. [Implementation Details](#implementation-details)

## Overview

CoRefusion combines three key components to address code refactoring localization:

1. **Pre-trained Code Encoder**: Extracts semantic representations from source code
2. **Discrete Diffusion Model**: Models the distribution of refactoring locations
3. **Localization Decoder**: Produces fine-grained predictions

### Key Innovations

- **Probabilistic Approach**: Uses diffusion models for uncertainty quantification
- **Multi-level Representations**: Combines token, line, and function-level information
- **Conditional Generation**: Conditions on rich code context for accurate predictions

## Problem Formulation

### Task Definition

Given a source code file $C$ consisting of $n$ lines:

$$C = \{l_1, l_2, \ldots, l_n\}$$

The task is to predict a binary label $y_i \in \{0, 1\}$ for each line $l_i$, where:

- $y_i = 1$ indicates that line $l_i$ requires refactoring
- $y_i = 0$ indicates that line $l_i$ does not require refactoring

### Input Representation

Code is represented at multiple levels:

1. **Token Level**: $\{t_1, t_2, \ldots, t_m\}$ - sequence of code tokens
2. **Line Level**: $\{l_1, l_2, \ldots, l_n\}$ - lines of code
3. **Structural Level**: Abstract Syntax Tree (AST) representation
4. **Semantic Level**: Learned embeddings from pre-trained models

### Output Format

The model produces:

- **Probability Distribution**: $P(y_i = 1 | C)$ for each line
- **Confidence Score**: Uncertainty estimate for each prediction
- **Contextual Information**: Attention weights showing relevant context

## Model Architecture

### 1. Code Encoder

The code encoder transforms source code into rich representations.

#### Pre-trained Base Model

We use CodeBERT (or similar) as the base encoder:

```python
encoder = AutoModel.from_pretrained("microsoft/codebert-base")
```

#### Multi-level Encoding

**Token-level encoding:**

$$\mathbf{h}_i^{\text{token}} = \text{Encoder}(t_i, C)$$

**Line-level aggregation:**

$$\mathbf{h}_j^{\text{line}} = \text{Pool}(\{\mathbf{h}_i^{\text{token}} : t_i \in l_j\})$$

**Function-level encoding:**

$$\mathbf{h}_k^{\text{func}} = \text{Attention}(\{\mathbf{h}_j^{\text{line}} : l_j \in f_k\})$$

#### Structural Features

Additional features capture code structure:

- **Complexity Metrics**: Cyclomatic complexity, LOC
- **Dependency Information**: Import statements, function calls
- **Nesting Depth**: Control flow nesting level
- **Code Patterns**: Common idioms and patterns

These are encoded and added to the base representations:

$$\mathbf{h}_i = \mathbf{h}_i^{\text{token}} + \text{FeatureEncoder}(\text{features}_i)$$

### 2. Discrete Diffusion Model

The diffusion model learns to predict refactoring locations through an iterative denoising process.

#### Forward Process

The forward process gradually adds noise to the labels:

$$q(y_t | y_{t-1}) = \text{Cat}(y_t; \mathbf{Q}_t y_{t-1})$$

where $\mathbf{Q}_t$ is a transition matrix that determines how labels become corrupted.

For discrete diffusion, we use:

$$\mathbf{Q}_t = (1 - \beta_t)\mathbf{I} + \beta_t / K \mathbf{1}\mathbf{1}^T$$

where:
- $\beta_t$ is the noise schedule
- $K$ is the number of classes (2 for binary)
- $\mathbf{I}$ is the identity matrix
- $\mathbf{1}$ is a vector of ones

#### Reverse Process

The reverse process learns to denoise:

$$p_\theta(y_{t-1} | y_t, C) = \text{Cat}(y_{t-1}; \mathbf{p}_\theta(y_t, C, t))$$

The denoising network predicts the clean labels:

$$\mathbf{p}_\theta(y_t, C, t) = \text{DenoiseNet}(y_t, \mathbf{h}^{\text{code}}, t)$$

#### Denoising Network Architecture

The denoising network consists of:

1. **Input Projection**: Project noisy labels to hidden dimension
2. **Time Embedding**: Sinusoidal embedding of timestep $t$
3. **Transformer Layers**: Process with self-attention
4. **Condition Integration**: Add code embeddings
5. **Output Projection**: Project to class logits

```python
class DenosingNetwork(nn.Module):
    def forward(self, y_t, code_emb, t):
        # Project input
        x = self.input_proj(y_t)
        
        # Add time and condition
        x = x + self.time_emb(t) + code_emb
        
        # Transform
        x = self.transformer(x)
        
        # Output
        return self.output_proj(x)
```

### 3. Localization Decoder

The decoder produces final predictions with confidence scores.

#### Hierarchical Attention

Multi-head attention aggregates information:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### Confidence Estimation

Confidence is estimated from:

1. **Prediction Entropy**: Lower entropy = higher confidence
2. **Diffusion Trajectory**: Stability across timesteps
3. **Attention Weights**: Agreement across attention heads

$$\text{Confidence} = 1 - H(\mathbf{p}_\theta(y_0 | C))$$

where $H$ is the entropy function.

## Training Procedure

### Two-Stage Training

#### Stage 1: Encoder Pre-training

Fine-tune the code encoder on auxiliary tasks:

- **Masked Language Modeling**: Predict masked tokens
- **Code Classification**: Classify code properties
- **Contrastive Learning**: Learn similar/dissimilar code pairs

#### Stage 2: End-to-End Training

Train the full model with the diffusion objective:

$$\mathcal{L} = \mathbb{E}_{q(y_{1:T}|y_0)} \left[ \sum_{t=1}^T D_{KL}(q(y_{t-1}|y_t, y_0) \| p_\theta(y_{t-1}|y_t, C)) \right]$$

In practice, we use a simplified objective:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, y_0, y_t} \left[ \| y_0 - \hat{y}_0(y_t, C, t) \|^2 \right]$$

### Optimization

**Optimizer**: AdamW with parameters:
- Learning rate: $5 \times 10^{-5}$
- Weight decay: $0.01$
- $\beta_1 = 0.9$, $\beta_2 = 0.999$

**Learning Rate Schedule**: Linear warmup followed by linear decay

**Gradient Clipping**: Max norm = 1.0

**Batch Size**: 32 samples per GPU

**Mixed Precision**: FP16 training with automatic mixed precision

### Data Augmentation

To improve robustness:

1. **Variable Renaming**: Rename variables randomly
2. **Comment Modification**: Add/remove/modify comments
3. **Formatting Changes**: Adjust whitespace and formatting
4. **Code Reordering**: Reorder independent statements

### Regularization

- **Dropout**: 0.1 on attention and feed-forward layers
- **Label Smoothing**: Smooth hard labels by $\epsilon = 0.1$
- **Early Stopping**: Patience of 5 epochs

## Inference Process

### Sampling Procedure

At inference, we sample from the learned distribution:

1. **Initialize**: Start with random noise $y_T \sim \mathcal{N}(0, I)$
2. **Iterative Denoising**: For $t = T, T-1, \ldots, 1$:
   - Predict: $\hat{y}_{t-1} = p_\theta(y_{t-1} | y_t, C)$
   - Sample: $y_{t-1} \sim \text{Cat}(\hat{y}_{t-1})$
3. **Final Prediction**: $\hat{y}_0 = \arg\max_k p_\theta(y_0^{(k)} | C)$

### Fast Sampling

For faster inference, we can skip timesteps using DDIM:

$$y_{t-\Delta t} = \sqrt{\alpha_{t-\Delta t}} \hat{y}_0(y_t, C, t) + \sqrt{1 - \alpha_{t-\Delta t}} \epsilon$$

This allows reducing from 1000 steps to 50-100 steps with minimal quality loss.

### Post-processing

After sampling, apply post-processing:

1. **Thresholding**: Apply confidence threshold
2. **Smoothing**: Merge adjacent predictions
3. **Filtering**: Remove isolated predictions
4. **Ranking**: Rank by confidence for prioritization

## Implementation Details

### Memory Optimization

- **Gradient Checkpointing**: Recompute activations during backward pass
- **Gradient Accumulation**: Accumulate gradients over multiple mini-batches
- **Model Parallelism**: Split model across multiple GPUs if needed

### Computational Efficiency

- **Efficient Attention**: Use Flash Attention or similar optimizations
- **Caching**: Cache encoder outputs for validation
- **Batching**: Dynamic batching for variable-length inputs
- **Quantization**: Use INT8 quantization for inference

### Reproducibility

To ensure reproducibility:

1. **Seed Everything**: Set all random seeds
2. **Deterministic Operations**: Use deterministic algorithms
3. **Version Control**: Pin all dependency versions
4. **Configuration Management**: Use Hydra for configs
5. **Experiment Tracking**: Log all hyperparameters and metrics

### Debugging Tips

1. **Sanity Checks**: 
   - Overfit on small dataset
   - Check gradient flow
   - Verify loss decreases

2. **Visualization**:
   - Plot attention weights
   - Visualize embeddings
   - Track metrics over time

3. **Profiling**:
   - Profile memory usage
   - Identify bottlenecks
   - Optimize hot paths

## Advanced Techniques

### Multi-Task Learning

Train jointly on related tasks:

- Code smell detection
- Bug localization
- Code clone detection

### Active Learning

Iteratively select informative samples:

1. Train on initial dataset
2. Predict on unlabeled pool
3. Select high-uncertainty samples
4. Get labels and retrain

### Domain Adaptation

Adapt to new domains:

- Fine-tune on domain-specific data
- Use domain adversarial training
- Apply meta-learning techniques

## Evaluation Considerations

### Metrics Selection

Choose metrics appropriate for the task:

- **Precision/Recall**: When false positives/negatives have different costs
- **F1-Score**: For balanced evaluation
- **MAP/NDCG**: When ranking matters
- **ROC-AUC**: For threshold-independent evaluation

### Statistical Testing

Always perform statistical tests:

- **Paired t-test**: Compare with baselines
- **Bonferroni correction**: For multiple comparisons
- **Effect size**: Report Cohen's d
- **Confidence intervals**: Show uncertainty

### Cross-Validation

Use appropriate validation strategies:

- **K-fold**: For limited data
- **Stratified**: Maintain class balance
- **Temporal**: For time-series data
- **Cross-project**: For generalization

## Common Pitfalls

1. **Data Leakage**: Ensure proper train/test split
2. **Overfitting**: Monitor validation metrics
3. **Hyperparameter Tuning**: Use separate validation set
4. **Class Imbalance**: Apply appropriate sampling/weighting
5. **Computational Resources**: Plan for training time and costs

## References

See `thesis/references.bib` for complete bibliography.

## Contact

For questions or clarifications:
- Email: [your.email@university.edu]
- GitHub Issues: [repository]/issues
