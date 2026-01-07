# Practice Problems: Lesson 8 - Neural Network Architectures

**Source:** Lessons/Lesson_8.md
**Subject Area:** AI Learning - Neural Network Architectures: Design Patterns and Modern Innovations
**Date Generated:** 2026-01-08
**Total Problems:** 5
**Estimated Completion Time:** 90-120 minutes

---

## Problem Distribution

| Type | Count | Difficulty | Focus |
|------|-------|------------|-------|
| Warm-Up | 1 | Foundation | Direct concept application |
| Skill-Builder | 2 | Intermediate | Multi-step procedures |
| Challenge | 1 | Advanced | Complex synthesis |
| Debug/Fix | 1 | Diagnostic | Error identification |

---

## Problem 1: Warm-Up - Parameter Count Calculation

**Difficulty:** Foundation
**Estimated Time:** 15 minutes
**Concepts:** Linear layers, convolution parameters, architecture analysis

### Problem Statement

Calculate the total number of trainable parameters for each of the following network components:

**A) MLP Block:**
```
Input: 768 dimensions
Linear(768, 3072) → GELU → Linear(3072, 768)
```

**B) Convolutional Block:**
```
Input: 64 channels, spatial size H×W
Conv2d(64, 128, kernel_size=3, padding=1)
BatchNorm2d(128)
Conv2d(128, 128, kernel_size=3, padding=1)
```

**C) Multi-Head Attention:**
```
Input: 512 dimensions
num_heads: 8
head_dim: 64 (512 / 8)
Wq, Wk, Wv, Wo projections
```

### Hints

<details>
<summary>Hint 1 (Linear Layer)</summary>
Linear(in, out) has: in × out weights + out biases = in × out + out parameters
</details>

<details>
<summary>Hint 2 (Convolution)</summary>
Conv2d(in_ch, out_ch, k×k) has: in_ch × out_ch × k × k + out_ch parameters
</details>

<details>
<summary>Hint 3 (Attention)</summary>
Each of Wq, Wk, Wv, Wo is a Linear(d, d) projection
</details>

### Solution

**A) MLP Block Parameters:**

```
Linear(768, 3072):
  Weights: 768 × 3072 = 2,359,296
  Biases:  3072
  Total:   2,362,368

GELU: 0 (no parameters)

Linear(3072, 768):
  Weights: 3072 × 768 = 2,359,296
  Biases:  768
  Total:   2,360,064

MLP Total: 2,362,368 + 2,360,064 = 4,722,432 parameters
```

**B) Convolutional Block Parameters:**

```
Conv2d(64, 128, 3×3):
  Weights: 64 × 128 × 3 × 3 = 73,728
  Biases:  128
  Total:   73,856

BatchNorm2d(128):
  Gamma (scale): 128
  Beta (shift):  128
  Total:         256 (trainable)
  Note: Running mean/var are buffers, not parameters

Conv2d(128, 128, 3×3):
  Weights: 128 × 128 × 3 × 3 = 147,456
  Biases:  128
  Total:   147,584

Conv Block Total: 73,856 + 256 + 147,584 = 221,696 parameters
```

**C) Multi-Head Attention Parameters:**

```
Wq (Query projection):  Linear(512, 512) = 512 × 512 + 512 = 262,656
Wk (Key projection):    Linear(512, 512) = 512 × 512 + 512 = 262,656
Wv (Value projection):  Linear(512, 512) = 512 × 512 + 512 = 262,656
Wo (Output projection): Linear(512, 512) = 512 × 512 + 512 = 262,656

MHA Total: 4 × 262,656 = 1,050,624 parameters
```

**Summary Table:**

| Component | Parameters | Notes |
|-----------|------------|-------|
| MLP Block | 4,722,432 | Expansion factor 4× is typical |
| Conv Block | 221,696 | Much smaller than MLP for same dims |
| MHA | 1,050,624 | 4d² for d-dimensional attention |

---

## Problem 2: Skill-Builder - Receptive Field Analysis

**Difficulty:** Intermediate
**Estimated Time:** 25 minutes
**Concepts:** Receptive field, convolution stacking, architectural design

### Problem Statement

You are designing a CNN for image classification where the network needs to "see" at least a 64×64 pixel region to make accurate predictions.

**Given architecture pattern:**
```
Each block: Conv2d(kernel=3, stride=1, padding=1) → ReLU → MaxPool(2×2, stride=2)
```

**Tasks:**

a) Calculate the receptive field after 1, 2, 3, and 4 blocks of this pattern.

b) How many blocks are needed to achieve a receptive field of at least 64×64?

c) If we use dilated convolutions with dilation=2 instead of standard convolutions in blocks 3 and 4, how does this change the answer?

d) Compare this approach to using a single large kernel (e.g., 7×7 or 11×11). What are the tradeoffs?

### Hints

<details>
<summary>Hint 1 (Receptive Field Formula)</summary>
RF_new = RF_old + (kernel_size - 1) × stride_product
Where stride_product is the product of all strides in previous layers.
</details>

<details>
<summary>Hint 2 (MaxPool Effect)</summary>
MaxPool doesn't add to receptive field directly, but it changes the effective stride for subsequent layers.
</details>

<details>
<summary>Hint 3 (Dilated Convolution)</summary>
Dilated convolution with dilation d has effective kernel size: k + (k-1)(d-1)
For k=3, d=2: effective size = 3 + 2×1 = 5
</details>

### Solution

**Part a) Receptive Field Calculation**

For each layer, RF grows based on kernel size and cumulative stride:

```
Initial RF = 1

Block 1:
  Conv3×3 (stride 1): RF = 1 + (3-1)×1 = 3
  MaxPool2×2 (stride 2): RF = 3 (unchanged, but stride doubles)
  Cumulative stride: 2

Block 2:
  Conv3×3: RF = 3 + (3-1)×2 = 7
  MaxPool2×2: RF = 7
  Cumulative stride: 4

Block 3:
  Conv3×3: RF = 7 + (3-1)×4 = 15
  MaxPool2×2: RF = 15
  Cumulative stride: 8

Block 4:
  Conv3×3: RF = 15 + (3-1)×8 = 31
  MaxPool2×2: RF = 31
  Cumulative stride: 16
```

| Blocks | Receptive Field | Cumulative Stride |
|--------|-----------------|-------------------|
| 1 | 3×3 | 2 |
| 2 | 7×7 | 4 |
| 3 | 15×15 | 8 |
| 4 | 31×31 | 16 |

**Part b) Blocks Needed for 64×64 RF**

Continuing the pattern:
```
Block 5:
  Conv3×3: RF = 31 + (3-1)×16 = 63
  Cumulative stride: 32

Block 6:
  Conv3×3: RF = 63 + (3-1)×32 = 127
  Cumulative stride: 64
```

**Answer: 6 blocks** are needed to achieve RF ≥ 64×64.

Actually, let's verify:
- After 5 blocks: RF = 63×63 (just under 64)
- After 6 blocks: RF = 127×127 (exceeds 64)

So **6 blocks** minimum for 64×64 RF.

**Part c) Dilated Convolutions in Blocks 3-4**

With dilation=2 in Conv3×3, effective kernel size = 5:

```
Block 1: RF = 3, stride = 2
Block 2: RF = 7, stride = 4

Block 3 (dilated, d=2):
  Effective kernel = 5
  RF = 7 + (5-1)×4 = 23
  Stride = 8

Block 4 (dilated, d=2):
  RF = 23 + (5-1)×8 = 55
  Stride = 16

Block 5:
  RF = 55 + (3-1)×16 = 87 (standard conv)
```

With dilation: **5 blocks** achieve RF > 64×64 vs 6 blocks without.

**Part d) Large Kernel Tradeoffs**

| Approach | Pros | Cons |
|----------|------|------|
| **Stacked 3×3** | Fewer params per layer, more non-linearity, well-studied | Many layers needed, slow training |
| **Large 7×7** | Fewer layers needed, larger RF quickly | More params (49 vs 9), less non-linearity |
| **Dilated 3×3** | Large RF, same params as 3×3, sparse sampling | Gaps in coverage, gridding artifacts |

**Calculation Example:**
```
3 stacked 3×3: 3 × (3×3×C²) = 27C² params, RF = 7×7
1 large 7×7:   1 × (7×7×C²) = 49C² params, RF = 7×7

Stacked 3×3 is 1.8× more parameter efficient for same RF.
```

**Modern approach (ConvNeXt):** Use 7×7 depthwise convolutions (much fewer params) combined with 1×1 pointwise convolutions for channel mixing.

---

## Problem 3: Skill-Builder - Attention Complexity Analysis

**Difficulty:** Intermediate
**Estimated Time:** 25 minutes
**Concepts:** Attention complexity, memory efficiency, architectural tradeoffs

### Problem Statement

You are optimizing a Transformer model for different deployment scenarios. Analyze the memory and compute requirements.

**Base Model Specifications:**
- Hidden dimension d = 1024
- Number of heads h = 16
- Head dimension d_k = 64
- Sequence length n (variable)

**Tasks:**

a) Calculate the memory required for storing the attention matrix for sequence lengths n = 1K, 4K, 16K, 64K tokens.

b) If your GPU has 24GB memory and the model weights use 4GB, what is the maximum sequence length you can process with standard attention?

c) With Flash Attention (which doesn't materialize the full attention matrix), how does this change?

d) Compare memory requirements for Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA with 4 groups) for a 32-layer model at n=4K.

### Hints

<details>
<summary>Hint 1 (Attention Memory)</summary>
Standard attention materializes an n×n matrix per head.
Total: batch × heads × n × n × bytes_per_element
</details>

<details>
<summary>Hint 2 (Flash Attention)</summary>
Flash Attention computes attention in tiles without storing full n×n matrix.
Memory: O(n) instead of O(n²)
</details>

<details>
<summary>Hint 3 (KV Cache)</summary>
MHA: stores separate K, V for each head
MQA: shares K, V across all heads
GQA: groups share K, V
</details>

### Solution

**Part a) Attention Matrix Memory**

For single head: n × n attention scores
For all heads: n × n × h values
Assuming FP16 (2 bytes per value), batch size 1:

```
Memory = n² × h × 2 bytes = n² × 16 × 2 = 32n² bytes
```

| Sequence Length | Memory Calculation | Memory Required |
|-----------------|-------------------|-----------------|
| n = 1K | 32 × (1,000)² | 32 MB |
| n = 4K | 32 × (4,000)² | 512 MB |
| n = 16K | 32 × (16,000)² | 8.2 GB |
| n = 64K | 32 × (64,000)² | 131 GB |

**Part b) Maximum Sequence Length (Standard Attention)**

Available memory for attention: 24GB - 4GB = 20GB = 20 × 10⁹ bytes

```
32n² ≤ 20 × 10⁹
n² ≤ 6.25 × 10⁸
n ≤ 25,000 tokens (approximately)
```

But this doesn't account for:
- Activations from other layers
- Optimizer states
- KV cache
- Intermediate values

**Practical limit:** ~10K-15K tokens with other overhead.

**Part c) Flash Attention Memory**

Flash Attention memory is O(n) not O(n²):

```
Memory ≈ O(n × d × h) for intermediate states
       ≈ n × 1024 × 2 bytes per layer
       ≈ 2n KB per layer
```

For n = 64K: ~128MB per layer (vs 131GB for standard!)

With Flash Attention, the limit becomes:
- Model weights: 4GB
- Activations: O(n × d × layers) ≈ linear in n
- Maximum n: limited by embedding tables and output layers, not attention

**Practical Flash Attention limit on 24GB:** Can handle 100K+ tokens.

**Part d) MHA vs MQA vs GQA Memory (KV Cache)**

At inference, we cache K, V for previous tokens:

```
KV Cache per layer per token: 2 × d_k × num_kv_heads × 2 bytes

MHA (16 heads):  2 × 64 × 16 × 2 = 4,096 bytes/token/layer
MQA (1 kv head): 2 × 64 × 1 × 2 = 256 bytes/token/layer
GQA (4 groups):  2 × 64 × 4 × 2 = 1,024 bytes/token/layer
```

For 32 layers, n = 4K tokens:

| Method | Per Layer | 32 Layers | Total |
|--------|-----------|-----------|-------|
| MHA | 4K × 4KB = 16MB | 512MB | **512MB** |
| MQA | 4K × 256B = 1MB | 32MB | **32MB** |
| GQA-4 | 4K × 1KB = 4MB | 128MB | **128MB** |

**Summary:**
- MQA: 16× smaller KV cache than MHA
- GQA-4: 4× smaller than MHA, 4× larger than MQA
- GQA provides quality between MHA and MQA with intermediate memory

---

## Problem 4: Challenge - Architecture Design for Multi-Scale Processing

**Difficulty:** Advanced
**Estimated Time:** 30 minutes
**Concepts:** U-Net, skip connections, hierarchical features, architecture design

### Problem Statement

Design a neural network architecture for satellite image analysis that must:

1. Process 1024×1024 input images
2. Detect objects of varying sizes (from 8×8 to 512×512 pixels)
3. Output both:
   - Classification (10 land-use categories)
   - Dense segmentation map (per-pixel labels)
4. Run inference in <100ms on an A100 GPU
5. Total parameters <50M

**Provide:**
- Complete architecture specification with layer details
- Explanation of how each requirement is addressed
- Parameter count calculation
- Justification for key design decisions

### Hints

<details>
<summary>Hint 1 (Multi-Scale)</summary>
Use encoder-decoder with skip connections (U-Net style) to capture features at multiple scales.
</details>

<details>
<summary>Hint 2 (Efficiency)</summary>
Consider depthwise separable convolutions, bottleneck blocks, or efficient attention for computational efficiency.
</details>

<details>
<summary>Hint 3 (Dual Output)</summary>
Classification can use global pooling on bottleneck features. Segmentation needs full resolution output.
</details>

### Solution

**Complete Architecture: EfficientUNet-Sat**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ENCODER (Feature Pyramid)                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Input: 1024×1024×3                                                       │
│      ↓                                                                    │
│  Stage 1: ConvBlock(3→32, stride 2) → 512×512×32      ─────────┐ Skip1   │
│      ↓                                                          │         │
│  Stage 2: EfficientBlock(32→64, stride 2) → 256×256×64 ────────┐│ Skip2  │
│      ↓                                                          ││        │
│  Stage 3: EfficientBlock(64→128, stride 2) → 128×128×128 ──────┐││ Skip3 │
│      ↓                                                          │││       │
│  Stage 4: EfficientBlock(128→256, stride 2) → 64×64×256 ───────┐│││Skip4 │
│      ↓                                                          ││││      │
│  Stage 5: EfficientBlock(256→512, stride 2) → 32×32×512        ││││      │
│                                                                  ││││      │
└──────────────────────────────────────────────────────────────────┼┼┼┼──────┘
                                                                   ││││
┌──────────────────────────────────────────────────────────────────┼┼┼┼──────┐
│                         BOTTLENECK                               ││││      │
├──────────────────────────────────────────────────────────────────┼┼┼┼──────┤
│  32×32×512 → SelfAttention(heads=8) → 32×32×512                  ││││      │
│      │                                                           ││││      │
│      ├──→ GlobalAvgPool → FC(512→10) → Classification Output     ││││      │
│      ↓                                                           ││││      │
└──────────────────────────────────────────────────────────────────┼┼┼┼──────┘
                                                                   ││││
┌──────────────────────────────────────────────────────────────────┼┼┼┼──────┐
│                         DECODER (Upsampling)                     ││││      │
├──────────────────────────────────────────────────────────────────┼┼┼┼──────┤
│  32×32×512 ↑ Upsample → Concat(Skip4) → EfficientBlock → 64×64×256  ◄┘│││  │
│      ↓                                                                │││  │
│  64×64×256 ↑ Upsample → Concat(Skip3) → EfficientBlock → 128×128×128 ◄┘││  │
│      ↓                                                                 ││  │
│  128×128×128 ↑ Upsample → Concat(Skip2) → EfficientBlock → 256×256×64 ◄┘│  │
│      ↓                                                                  │  │
│  256×256×64 ↑ Upsample → Concat(Skip1) → ConvBlock → 512×512×32        ◄┘  │
│      ↓                                                                      │
│  512×512×32 ↑ Upsample → Conv(32→10) → 1024×1024×10 → Segmentation Output  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Detailed Block Specifications:**

**EfficientBlock (Inverted Residual with SE):**
```python
class EfficientBlock(in_ch, out_ch, stride=1, expand_ratio=4):
    # Expansion
    Conv1x1(in_ch, in_ch * expand_ratio)
    BatchNorm, SiLU

    # Depthwise
    DepthwiseConv3x3(in_ch * expand_ratio, stride=stride)
    BatchNorm, SiLU

    # Squeeze-and-Excitation
    SE(in_ch * expand_ratio, reduction=4)

    # Projection
    Conv1x1(in_ch * expand_ratio, out_ch)
    BatchNorm

    # Residual (if stride=1 and in_ch==out_ch)
    if stride == 1 and in_ch == out_ch:
        output = input + projected
```

**Parameter Count Calculation:**

```
Encoder:
  Stage 1: 3×32×3×3 + 32 = 896
  Stage 2: EfficientBlock(32→64) ≈ 15K
  Stage 3: EfficientBlock(64→128) ≈ 50K
  Stage 4: EfficientBlock(128→256) ≈ 180K
  Stage 5: EfficientBlock(256→512) ≈ 700K
Encoder total: ~950K

Bottleneck Attention:
  Q, K, V, O projections: 4 × 512² = 1.05M
  Total: ~1.1M

Decoder (mirrors encoder):
  ~950K parameters

Classification Head:
  FC(512→10): 5,130

Segmentation Head:
  Conv(32→10): 2,890

Total: ~3.0M parameters
```

**Wait, this is well under 50M. Let's scale up for better accuracy:**

**Scaled Architecture (meeting constraints):**

```
Encoder channels: [64, 128, 256, 512, 1024]
Decoder channels: [512, 256, 128, 64, 32]
Add 2 EfficientBlocks per stage

Revised parameter count:
  Encoder: ~15M
  Bottleneck (with 2 attention layers): ~4M
  Decoder: ~15M
  Heads: ~0.5M

Total: ~34.5M parameters (under 50M limit)
```

**How Requirements Are Addressed:**

| Requirement | Solution |
|-------------|----------|
| 1024×1024 input | Initial stride-2 conv reduces to 512, manageable sizes |
| Multi-scale (8-512px) | 5-level pyramid: 32×32 to 512×512 feature maps |
| Classification | Global pooling on 32×32 bottleneck |
| Segmentation | U-Net decoder with skip connections |
| <100ms inference | Efficient blocks (depthwise), moderate channels |
| <50M params | EfficientNet-style blocks, no redundancy |

**Latency Estimation:**

```
Major operations on A100:
- Encoder convolutions: ~15ms
- Bottleneck attention (32×32=1K tokens): ~5ms
- Decoder convolutions: ~15ms
- Upsampling: ~5ms

Estimated total: ~40ms (well under 100ms)
```

**Key Design Decisions:**

1. **EfficientBlock over standard Conv:** 3× fewer parameters with comparable expressiveness

2. **Attention only at bottleneck:** 32×32=1K tokens is manageable; earlier would be too expensive

3. **Skip connections at all scales:** Essential for small object detection (8×8 objects visible at 512×512 feature map)

4. **Dual output heads:** Share all computation, only differ at final layer

---

## Problem 5: Debug/Fix - Architecture Errors

**Difficulty:** Diagnostic
**Estimated Time:** 20 minutes
**Concepts:** Common architecture mistakes, dimension mismatches, design flaws

### Problem Statement

Review the following architecture descriptions and identify the errors. For each, explain the problem and provide the fix.

**Scenario A: Transformer Encoder**
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x + self.attn(x, x, x))
        x = self.norm(x + self.ffn(x))
        return x
```

**Scenario B: ResNet Block**
```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out
```

**Scenario C: LSTM Configuration**
```python
model = nn.Sequential(
    nn.Embedding(vocab_size, 256),
    nn.LSTM(256, 512, num_layers=3, batch_first=True),
    nn.Linear(512, vocab_size)
)
```

**Scenario D: CNN Classifier**
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),   # 224 -> 222
            nn.ReLU(),
            nn.MaxPool2d(2),       # 222 -> 111
            nn.Conv2d(64, 128, 3), # 111 -> 109
            nn.ReLU(),
            nn.MaxPool2d(2),       # 109 -> 54
        )
        self.classifier = nn.Linear(128 * 54 * 54, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

### Hints

<details>
<summary>Hint A</summary>
Compare Pre-LN vs Post-LN Transformer. Also check the activation function choice for modern Transformers.
</details>

<details>
<summary>Hint B</summary>
What happens if in_ch ≠ out_ch? Can you add tensors of different shapes?
</details>

<details>
<summary>Hint C</summary>
LSTM returns a tuple (output, (hidden, cell)). nn.Sequential doesn't handle this properly.
</details>

<details>
<summary>Hint D</summary>
MaxPool2d(2) on odd dimensions rounds down. Check: 111 // 2 = ?
</details>

### Solution

**Scenario A: Transformer Encoder - Issues & Fixes**

**Problem 1: Post-LN instead of Pre-LN**
```python
# Current (Post-LN): x = norm(x + sublayer(x))
# This is less stable for training deep Transformers
```

**Problem 2: ReLU instead of GELU**
```python
# ReLU is outdated for Transformers; GELU is standard
```

**Problem 3: Single LayerNorm for both sublayers**
```python
# Should have separate LayerNorm for attention and FFN
```

**Fixed Code:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),  # Fix: GELU instead of ReLU
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)  # Fix: Separate norms
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Fix: Pre-LN (norm before sublayer)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

---

**Scenario B: ResNet Block - Issues & Fixes**

**Problem: Dimension mismatch when in_ch ≠ out_ch**
```python
# If in_ch=64, out_ch=128:
# residual has shape [B, 64, H, W]
# out has shape [B, 128, H, W]
# Can't add them!
```

**Fixed Code:**
```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Fix: Add projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.shortcut(x)  # Fix: Project if needed
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out
```

---

**Scenario C: LSTM Configuration - Issues & Fixes**

**Problem: LSTM returns tuple, breaks nn.Sequential**
```python
# nn.LSTM returns (output, (h_n, c_n))
# nn.Sequential passes full tuple to next layer
# nn.Linear expects tensor, not tuple
```

**Fixed Code:**
```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 512, num_layers=3, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)  # Fix: Unpack tuple
        # Option 1: Use last hidden state
        out = self.fc(h_n[-1])
        # Option 2: Use all outputs (for seq2seq)
        # out = self.fc(lstm_out)
        return out
```

---

**Scenario D: CNN Classifier - Issues & Fixes**

**Problem: Incorrect spatial dimension calculation**
```python
# 224 - 2 = 222 (no padding on 3×3 conv)
# 222 / 2 = 111
# 111 - 2 = 109
# 109 / 2 = 54.5 → 54 (floor)  # CORRECT

# But the real issue is:
# The dimensions in Linear layer are wrong!
# After features: [B, 128, 54, 54]
# 128 * 54 * 54 = 373,248 (correct)

# Actually, let's recalculate more carefully:
# Conv(3): 224 → 222 (loses 2)
# Pool(2): 222 → 111
# Conv(3): 111 → 109
# Pool(2): 109 → 54 (floor division)

# Wait, the Linear dim looks correct. Let me check for other issues...
```

**Actual Problems:**

1. **No padding causes dimension loss** (may be intentional but fragile)
2. **Missing BatchNorm** (not an error but suboptimal)
3. **Hardcoded dimensions** (breaks if input size changes)

**Better Fixed Code:**
```python
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   # 224 -> 224 (with padding)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 224 -> 112
            nn.Conv2d(64, 128, 3, padding=1), # 112 -> 112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 112 -> 56
        )
        # Fix: Use adaptive pooling to handle any input size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)       # [B, 128, 1, 1]
        x = x.view(x.size(0), -1) # [B, 128]
        return self.classifier(x)
```

**Key fixes:**
- Added padding to preserve dimensions
- Added BatchNorm for better training
- Used AdaptiveAvgPool to handle variable input sizes
- Cleaner, more robust architecture

---

## Self-Assessment Guide

| Level | Criteria |
|-------|----------|
| **Mastery** | Completed all problems correctly, identified multiple valid approaches |
| **Proficient** | Solved 4/5 problems, minor errors in complex scenarios |
| **Developing** | Solved warm-up and skill-builders, struggled with challenge/debug |
| **Foundational** | Need review of core architecture concepts |

---

*Generated from Lesson 8: Neural Network Architectures | Practice Problems Skill*
