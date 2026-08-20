# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Ngo Tri Si's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/ngotrisi2004.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Bigram Probabilities (Add-1 Smoothing) | Estimate bigram probabilities from token sequences using add-one smoothing over a fixed vocabulary. | https://www.tensortonic.com/problems/bigram-probabilities |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Dot Product | Implement a multi-block CUDA dot-product reduction that combines partial sums into one scalar output. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/dot-product |
| GELU | Implement exact GELU activation in CUDA with one thread per element and the device error-function intrinsic. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/gelu |
| Matrix-Vector Multiplication | Implement row-major CUDA matrix-vector multiplication with one thread computing each output row. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/gemv |
| Hadamard Product | Implement elementwise matrix multiplication in CUDA using a two-dimensional grid and row-major bounds-checked indexing. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/hadamard-product |
| Layer Normalization | Implement fused row-wise LayerNorm in CUDA with shared-memory mean and variance reduction, affine scale, and bias. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/layer-norm |
| Leaky ReLU | Implement Leaky ReLU activation in CUDA with one thread per element, bounds checks, and a configurable negative slope. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/leaky-relu |
| Matrix Addition | Implement elementwise matrix addition in CUDA with a two-dimensional grid, row-major indexing, and bounds checks. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/matrix-addition |
| Matrix Multiplication | Implement row-major matrix multiplication in CUDA with one thread per output element and inner-product accumulation. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/matrix-multiplication |
| Matrix Transpose | Implement matrix transpose in CUDA with a two-dimensional launch grid, row-major buffers, and bounds-checked writes. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/matrix-transpose |
| Outer Product | Compute a vector outer product in CUDA with a two-dimensional grid, row-major output, and bounds-checked indexing. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/outer-product |
| ReLU | Implement ReLU activation in CUDA with one thread per element, bounds checks, and branch-efficient rectification. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/relu |
| RMS Normalization | Implement row-wise RMS normalization in CUDA with sum-of-squares reduction, numerical stability, and learnable scaling. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/rms-norm |
| Sigmoid | Implement sigmoid activation in CUDA with one thread per element, device exponential math, and bounds-checked memory access. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/sigmoid |
| Softmax | Implement numerically stable CUDA softmax over a vector using global maximum and normalization reductions. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/softmax |
| Sum of Array | Implement a multi-block CUDA sum reduction that combines partial block sums into one scalar output. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/sum-of-array |
| Swish | Implement fused Swish or SiLU activation in CUDA with one thread per element and device exponential math. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/swish |
| Tanh | Implement hyperbolic tangent activation in CUDA with one thread per element, device intrinsic math, and bounds checks. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/tanh |
| Vector Addition | Implement bounds-checked pointwise vector addition in CUDA with one thread per output element. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/vector-addition |
| Vector Subtraction | Implement bounds-checked pointwise vector subtraction in CUDA with one thread per output element. | https://www.tensortonic.com/study-plans/cuda-basics/cuda/vector-subtract |
| Apply Ranked BPE Merges | Apply learned byte-pair merge rules to UTF-8 byte IDs in their supplied priority order, then reconstruct text through the supplied vocabulary. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l01-apply-bpe-merge-ranks |
| Train a Deterministic BPE Vocabulary | Choose the highest count with a lexicographic byte-string tie break, assign the next token ID, and replace non-overlapping matches from left to right. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l01-train-byte-pair-encoding |
| Named-Dimension Batched Attention Scores | Compute batched multi-head query-key scores by contracting only the head-width dimension. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-einsum-attention-scores |
| Gradient Accumulation Equivalence | Combine mean-loss gradients from unequal microbatches into one full-batch mean gradient, then apply a single SGD update. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-gradient-accumulation-step |
| Transformer Training FLOP Estimator | Estimate one training step from forward matrix multiplications and a supplied forward attention cost. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-training-flop-estimator |
| Mixed-Precision Training Memory Accountant | Compute exact storage for parameters, gradients, saved activations, and optimizer state from tensor shapes and byte widths. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l02-training-memory-accountant |
| Causal Grouped-Query Attention | Compute causal scaled dot-product attention in which each contiguous group of query heads shares one key/value head. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-causal-grouped-query-attention |
| Parameter-Matched SwiGLU Block | Choose a parameter-matched SwiGLU hidden width under an available-width limit, then evaluate the bias-free block. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-parameter-matched-swiglu |
| RMSNorm Forward Pass | Normalize each final-dimension vector by its root mean square and apply the learned scale without mean subtraction. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-rmsnorm-forward |
| Rotary Query and Key Embeddings | Rotate each adjacent coordinate pair of query and key vectors by a position-dependent angle. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l03-rotary-query-key-embeddings |
| Gated DeltaNet State Update | Decay the recurrent state, erase its component along a unit key, write the new value, and read the just-updated state. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l04-gated-deltanet-scan |
| Parallel and Recurrent Linear Attention | Compute causal softmax-free linear attention through both a parallel formulation and a recurrent state scan. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l04-linear-attention-duality |
| Mamba 2 Gated State Scan | Apply a gated recurrent state update and read each output from the just-updated state. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l04-mamba2-gated-state-scan |
| Top-k MoE Router with Load Statistics | Route each token to its highest-scoring experts, combine selected outputs, add the shared expert, and report load statistics. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l04-topk-moe-router |
| Global-Memory Coalescing Counter | Count the aligned cache lines touched by a warp's fixed-width global-memory accesses and measure useful transferred bytes. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l05-global-memory-coalescing |
| GPU Occupancy Calculator | Calculate resident blocks, resident warps, and occupancy from one block's resource use and one SM's limits. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l05-gpu-occupancy-calculator |
| Shared-Memory Bank Conflict Analyzer | Analyze GPU shared-memory addresses by warp, reporting bank indices and the conflict degree for each access step. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cs336-l05-shared-memory-bank-conflicts |
| Blockwise Online Softmax | Implement stable blockwise online softmax in CUDA for contiguous or strided rows across float32, float16, and bfloat16 inputs. | https://www.tensortonic.com/study-plans/language-modeling-from-scratch/cuda/cs336-l05-blockwise-online-softmax |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/ngotrisi2004)
<!-- tensortonic:end -->
