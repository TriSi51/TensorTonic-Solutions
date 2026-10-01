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
| Argmax | Implement a parallel CUDA argmax reduction that returns the lowest index when multiple elements share the maximum. | https://www.tensortonic.com/problems/argmax |
| Argmin | Implement a parallel CUDA argmin reduction that returns the lowest index when multiple elements share the minimum. | https://www.tensortonic.com/problems/argmin |
| Bigram Probabilities (Add-1 Smoothing) | Estimate bigram probabilities from token sequences using add-one smoothing over a fixed vocabulary. | https://www.tensortonic.com/problems/bigram-probabilities |
| Apply Ranked BPE Merges | Apply learned byte-pair merge rules to UTF-8 byte IDs in their supplied priority order, then reconstruct text through the supplied vocabulary. | https://www.tensortonic.com/problems/cs336-l01-apply-bpe-merge-ranks |
| Train a Deterministic BPE Vocabulary | Choose the highest count with a lexicographic byte-string tie break, assign the next token ID, and replace non-overlapping matches from left to right. | https://www.tensortonic.com/problems/cs336-l01-train-byte-pair-encoding |
| Named-Dimension Batched Attention Scores | Compute batched multi-head query-key scores by contracting only the head-width dimension. | https://www.tensortonic.com/problems/cs336-l02-einsum-attention-scores |
| Gradient Accumulation Equivalence | Combine mean-loss gradients from unequal microbatches into one full-batch mean gradient, then apply a single SGD update. | https://www.tensortonic.com/problems/cs336-l02-gradient-accumulation-step |
| Transformer Training FLOP Estimator | Estimate one training step from forward matrix multiplications and a supplied forward attention cost. | https://www.tensortonic.com/problems/cs336-l02-training-flop-estimator |
| Mixed-Precision Training Memory Accountant | Compute exact storage for parameters, gradients, saved activations, and optimizer state from tensor shapes and byte widths. | https://www.tensortonic.com/problems/cs336-l02-training-memory-accountant |
| Causal Grouped-Query Attention | Compute causal scaled dot-product attention in which each contiguous group of query heads shares one key/value head. | https://www.tensortonic.com/problems/cs336-l03-causal-grouped-query-attention |
| Parameter-Matched SwiGLU Block | Choose a parameter-matched SwiGLU hidden width under an available-width limit, then evaluate the bias-free block. | https://www.tensortonic.com/problems/cs336-l03-parameter-matched-swiglu |
| RMSNorm Forward Pass | Normalize each final-dimension vector by its root mean square and apply the learned scale without mean subtraction. | https://www.tensortonic.com/problems/cs336-l03-rmsnorm-forward |
| Rotary Query and Key Embeddings | Rotate each adjacent coordinate pair of query and key vectors by a position-dependent angle. | https://www.tensortonic.com/problems/cs336-l03-rotary-query-key-embeddings |
| Gated DeltaNet State Update | Decay the recurrent state, erase its component along a unit key, write the new value, and read the just-updated state. | https://www.tensortonic.com/problems/cs336-l04-gated-deltanet-scan |
| Parallel and Recurrent Linear Attention | Compute causal softmax-free linear attention through both a parallel formulation and a recurrent state scan. | https://www.tensortonic.com/problems/cs336-l04-linear-attention-duality |
| Mamba 2 Gated State Scan | Apply a gated recurrent state update and read each output from the just-updated state. | https://www.tensortonic.com/problems/cs336-l04-mamba2-gated-state-scan |
| Top-k MoE Router with Load Statistics | Route each token to its highest-scoring experts, combine selected outputs, add the shared expert, and report load statistics. | https://www.tensortonic.com/problems/cs336-l04-topk-moe-router |
| Blockwise Online Softmax | Implement stable blockwise online softmax in CUDA for contiguous or strided rows across float32, float16, and bfloat16 inputs. | https://www.tensortonic.com/problems/cs336-l05-blockwise-online-softmax |
| Global-Memory Coalescing Counter | Count the aligned cache lines touched by a warp's fixed-width global-memory accesses and measure useful transferred bytes. | https://www.tensortonic.com/problems/cs336-l05-global-memory-coalescing |
| GPU Occupancy Calculator | Calculate resident blocks, resident warps, and occupancy from one block's resource use and one SM's limits. | https://www.tensortonic.com/problems/cs336-l05-gpu-occupancy-calculator |
| Shared-Memory Bank Conflict Analyzer | Analyze GPU shared-memory addresses by warp, reporting bank indices and the conflict degree for each access step. | https://www.tensortonic.com/problems/cs336-l05-shared-memory-bank-conflicts |
| Dot Product | Implement a multi-block CUDA dot-product reduction that combines partial sums into one scalar output. | https://www.tensortonic.com/problems/cuda-dot-product |
| GELU | Implement exact GELU activation in CUDA with one thread per element and the device error-function intrinsic. | https://www.tensortonic.com/problems/cuda-gelu |
| Leaky ReLU | Implement Leaky ReLU activation in CUDA with one thread per element, bounds checks, and a configurable negative slope. | https://www.tensortonic.com/problems/cuda-leaky-relu |
| Matrix Transpose | Implement matrix transpose in CUDA with a two-dimensional launch grid, row-major buffers, and bounds-checked writes. | https://www.tensortonic.com/problems/cuda-matrix-transpose |
| CLIP Cosine Retrieval | Rank corpus embeddings for each CLIP-style query by stable cosine similarity, including safe handling of zero vectors. | https://www.tensortonic.com/problems/cv-clip-cosine-retrieval |
| Matrix-Vector Multiplication | Implement row-major CUDA matrix-vector multiplication with one thread computing each output row. | https://www.tensortonic.com/problems/gemv |
| Hadamard Product | Implement elementwise matrix multiplication in CUDA using a two-dimensional grid and row-major bounds-checked indexing. | https://www.tensortonic.com/problems/hadamard-product |
| Implement Autoregressive Decoding with a KV Cache | Decode autoregressively with an append-only key-value cache so previously processed tokens are never recomputed. | https://www.tensortonic.com/problems/inference-autoregressive-kv-cache |
| Implement FlashAttention with Online Softmax | Implement tiled attention with online softmax statistics that agrees numerically with dense scaled attention. | https://www.tensortonic.com/problems/inference-flash-attention-online-softmax |
| Implement Grouped-Query Attention (GQA) | Implement grouped-query attention by mapping query heads onto fewer key-value heads with validated head divisibility. | https://www.tensortonic.com/problems/inference-grouped-query-attention |
| Calculate KV Cache Memory for MHA, MQA, GQA, and MLA | Compute the total KV-cache memory, in bytes, for a full sequence under four attention variants: MHA, MQA, GQA, and MLA. | https://www.tensortonic.com/problems/inference-kv-cache-memory |
| Implement MoE Token Dispatch and Expert Aggregation | Run each expert's feed-forward computation only on the tokens routed to it, and scatter-add the routing-weighted results back into token order. | https://www.tensortonic.com/problems/inference-moe-dispatch-aggregation |
| Implement Sparse MoE Top-k Expert Routing | Select the top k highest-scoring experts per token and compute routing weights as a softmax over only those selected logits. | https://www.tensortonic.com/problems/inference-moe-top-k-routing |
| Implement Multi-Head Attention (MHA) | Split into h heads, run scaled dot-product attention per head with an optional causal mask, concatenate the heads, and apply an output projection. | https://www.tensortonic.com/problems/inference-multi-head-attention |
| Implement Multi-Head Latent Attention (MLA) | Implement Multi-Head Latent Attention (MLA), and return both the projected attention output and the compressed latent tensor. | https://www.tensortonic.com/problems/inference-multi-head-latent-attention |
| Implement Multi-Query Attention (MQA) | Implement multi-query attention with separate query heads, shared key-value heads, optional masking, and output projection. | https://www.tensortonic.com/problems/inference-multi-query-attention |
| Implement PagedAttention Block Allocation | Assign each sequence enough fixed-size blocks to hold its tokens, drawn from the free pool in order. | https://www.tensortonic.com/problems/inference-paged-attention-allocation |
| Implement Per-Channel Weight Quantization | Quantize weights independently along a selected channel axis, then dequantize with channel-specific scales. | https://www.tensortonic.com/problems/inference-per-channel-quantization |
| Implement Prefix Cache Matching and Reuse | Find the longest exactly-matching prefix, measured in whole blocks, among all candidates. | https://www.tensortonic.com/problems/inference-prefix-cache-reuse |
| Implement Rotary Position Embeddings for Decoding | Apply rotary position embeddings to query and key tensors at supplied absolute decoding positions. | https://www.tensortonic.com/problems/inference-rotary-position-embeddings |
| Implement Scaled Dot-Product Attention | Implement batched scaled dot-product attention for self- and cross-attention with optional masks and stable softmax. | https://www.tensortonic.com/problems/inference-scaled-dot-product-attention |
| Implement Speculative Decoding and Token Verification | Verify speculative draft tokens against target probabilities and sample the corrected or bonus continuation exactly. | https://www.tensortonic.com/problems/inference-speculative-decoding-verification |
| Implement Symmetric INT8 Quantization | Quantize finite tensors to symmetric INT8 values with an absolute-maximum scale and reconstruct dequantized outputs. | https://www.tensortonic.com/problems/inference-symmetric-int8-quantization |
| Implement Temperature, Top-k, and Top-p Sampling | Sample tokens from batched logits using temperature scaling, top-k filtering, top-p filtering, and deterministic randomness. | https://www.tensortonic.com/problems/inference-token-sampling |
| L1 Normalization | Implement CUDA L1 vector normalization by reducing absolute values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l1-normalize |
| L2 Normalization | Implement CUDA L2 vector normalization by reducing squared values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l2-normalize |
| Layer Normalization | Implement fused row-wise LayerNorm in CUDA with shared-memory mean and variance reduction, affine scale, and bias. | https://www.tensortonic.com/problems/layer-norm |
| Matrix Addition | Implement elementwise matrix addition in CUDA with a two-dimensional grid, row-major indexing, and bounds checks. | https://www.tensortonic.com/problems/matrix-addition |
| Matrix Multiplication | Implement row-major matrix multiplication in CUDA with one thread per output element and inner-product accumulation. | https://www.tensortonic.com/problems/matrix-multiplication |
| Max of Array | Implement a multi-block CUDA maximum reduction that combines block-local maxima into one scalar output. | https://www.tensortonic.com/problems/max-of-array |
| Mean and Variance | Compute population mean and variance in CUDA with device-side reductions and synchronized scalar outputs. | https://www.tensortonic.com/problems/mean-variance |
| Implement Micro-F1 | Compute multiclass micro-F1 by aggregating true positives, false positives, and false negatives across labels. | https://www.tensortonic.com/problems/metrics-f1-micro |
| Min of Array | Implement a multi-block CUDA minimum reduction that combines block-local minima into one scalar output. | https://www.tensortonic.com/problems/min-of-array |
| Outer Product | Compute a vector outer product in CUDA with a two-dimensional grid, row-major output, and bounds-checked indexing. | https://www.tensortonic.com/problems/outer-product |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Percentiles / Quantiles | Compute requested data percentiles with linear interpolation using NumPy-compatible quantile semantics. | https://www.tensortonic.com/problems/probstat-percentiles |
| ReLU | Implement ReLU activation in CUDA with one thread per element, bounds checks, and branch-efficient rectification. | https://www.tensortonic.com/problems/relu |
| RMS Normalization | Implement row-wise RMS normalization in CUDA with sum-of-squares reduction, numerical stability, and learnable scaling. | https://www.tensortonic.com/problems/rms-norm |
| Sigmoid | Implement sigmoid activation in CUDA with one thread per element, device exponential math, and bounds-checked memory access. | https://www.tensortonic.com/problems/sigmoid |
| Softmax | Implement numerically stable CUDA softmax over a vector using global maximum and normalization reductions. | https://www.tensortonic.com/problems/softmax |
| Sum of Array | Implement a multi-block CUDA sum reduction that combines partial block sums into one scalar output. | https://www.tensortonic.com/problems/sum-of-array |
| Swish | Implement fused Swish or SiLU activation in CUDA with one thread per element and device exponential math. | https://www.tensortonic.com/problems/swish |
| Tanh | Implement hyperbolic tangent activation in CUDA with one thread per element, device intrinsic math, and bounds checks. | https://www.tensortonic.com/problems/tanh |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/problems/triton-vector-addition |
| Vector Addition | Implement bounds-checked pointwise vector addition in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-addition |
| Vector Subtraction | Implement bounds-checked pointwise vector subtraction in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-subtract |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/ngotrisi2004)
<!-- tensortonic:end -->
