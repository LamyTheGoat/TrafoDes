# TrafoDes Deep Learning System - Learning Guide

## What Was Built: The Big Picture

I created a **deep learning layer** on top of your existing physics-based transformer optimizer. Think of it as teaching a neural network to "learn" your physics equations, so it can predict results 1000x faster.

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR EXISTING SYSTEM                         │
│   mainRect.py → Physics calculations → Grid search → Result     │
│   (5-60 seconds per optimization)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    NEW ML SYSTEM                                │
│   Neural Network → Instant prediction → Gradient optimization   │
│   (10-100 milliseconds per optimization)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Data Generation (`ml/data/generate_dataset.py`)

### What It Does
Generates training data by running your physics equations thousands of times with different inputs.

### How It Works
```
Step 1: Sample 100,000 random designs
        ↓
Step 2: For each design, run YOUR physics calculations
        - CalculateNoLoadLosses()
        - CalculateLoadLosses()
        - CalculateImpedance()
        - CalculatePrice()
        ↓
Step 3: Save (input, output) pairs to HDF5 file
```

### Key Concept: Latin Hypercube Sampling (LHS)
Instead of random sampling, LHS ensures we cover the entire design space evenly:

```
Random Sampling:          Latin Hypercube:
. .   .                   .     .     .
    . .  .                    .     .
  .    .                  .     .     .
      .  .                    .     .
```

### The Data Format
```
INPUTS (7 parameters):           OUTPUTS (what we predict):
─────────────────────           ────────────────────────
core_diameter    (80-500 mm)    nll      (No-Load Loss in W)
core_length      (0-500 mm)     ll       (Load Loss in W)
lv_turns         (5-100)        ucc      (Impedance in %)
foil_height      (200-1200 mm)  price    (Cost in $)
foil_thickness   (0.3-4 mm)     is_valid (True/False)
hv_thickness     (1-5 mm)
hv_length        (3.5-20 mm)
```

---

## Component 2: Surrogate Model (`ml/models/surrogate.py`)

### What It Does
A neural network that **mimics** your physics calculations but runs 1000x faster.

### Architecture Diagram
```
                    ┌─────────────────┐
                    │  7 Input Params │
                    │  (normalized)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         ┌────────┐    ┌────────┐    ┌────────┐
         │Dense   │    │Dense   │    │Dense   │
         │256     │───▶│256     │───▶│128     │───▶ Dense(64)
         │+ReLU   │    │+ReLU   │    │+ReLU   │     SHARED
         └────────┘    └────────┘    └────────┘     LAYERS
                                          │
              ┌───────────┬───────────┬───┴───┬───────────┐
              ▼           ▼           ▼       ▼           ▼
         ┌────────┐  ┌────────┐  ┌────────┐ ┌────────┐ ┌────────┐
         │NLL Head│  │LL Head │  │Ucc Head│ │Price   │ │Valid   │
         │32→1    │  │32→1    │  │32→1    │ │Head    │ │Head    │
         └────────┘  └────────┘  └────────┘ │32→1    │ │16→1    │
              │           │           │     └────────┘ │sigmoid │
              ▼           ▼           ▼          │     └────────┘
            NLL         LL          Ucc       Price      │
           (W)         (W)         (%)        ($)        ▼
                                                      Valid
                                                     (0-1)
```

### Why Multi-Task?
Instead of 5 separate models, we use ONE model with shared layers because:
- NLL, LL, Ucc, Price are all related to the same physics
- Shared layers learn common patterns (e.g., "bigger core = more losses")
- Training is more efficient

### Training Process
```python
# Pseudocode of training loop:
for epoch in range(100):
    for batch in training_data:
        # Forward pass
        predictions = model(batch.inputs)

        # Calculate loss (how wrong are we?)
        loss = MSE(predictions.nll, batch.true_nll) +
               MSE(predictions.ll, batch.true_ll) +
               MSE(predictions.ucc, batch.true_ucc) +
               MSE(predictions.price, batch.true_price) +
               BCE(predictions.valid, batch.true_valid)

        # Backward pass (adjust weights to reduce error)
        loss.backward()
        optimizer.step()
```

---

## Component 3: Inverse Design Model (`ml/models/inverse.py`)

### What It Does
The **reverse** of the surrogate: given target specs, predict optimal design.

```
NORMAL DIRECTION (Surrogate):
  Design Params → Neural Network → NLL, LL, Ucc, Price

INVERSE DIRECTION (This model):
  Target NLL, LL, Ucc → Neural Network → Optimal Design Params
```

### The Problem: One-to-Many Mapping
Many different designs can achieve the same specs:
```
Target: NLL=800W, LL=7000W, Ucc=6%

Valid Design A: core_dia=200, lv_turns=25, ...
Valid Design B: core_dia=180, lv_turns=30, ...
Valid Design C: core_dia=220, lv_turns=22, ...
```

### Solution: Conditional VAE (CVAE)
A special type of neural network that can generate **multiple valid solutions**:

```
┌─────────────┐
│ Target Specs│
│ (NLL,LL,Ucc)│
└──────┬──────┘
       │
       ▼
┌──────────────┐     ┌────────────────┐
│   ENCODER    │────▶│ Latent Space   │  ← Random sampling here
│ (Learn the   │     │ (compressed    │    generates different
│  pattern)    │     │  representation│    valid designs
└──────────────┘     └───────┬────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │    DECODER     │
                    │ (Generate      │
                    │  design)       │
                    └───────┬────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │ Design Params  │
                    │ (core_dia,     │
                    │  lv_turns,...) │
                    └────────────────┘
```

---

## Component 4: RL Environment (`ml/envs/transformer_env.py`)

### What It Does
A training environment for **Reinforcement Learning** agents to learn optimization.

### RL Concept Explained
```
Traditional Optimization:
  Try many designs → Pick the best one

RL Optimization:
  Agent LEARNS a STRATEGY for finding good designs
  (Like teaching someone to fish vs giving them fish)
```

### Environment Structure
```
┌─────────────────────────────────────────────────────────────┐
│                    RL ENVIRONMENT                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STATE (what the agent sees):                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Current 7 design parameters (normalized)           │   │
│  │ • Current cost                                       │   │
│  │ • Constraint violations (NLL, LL, Ucc over target)   │   │
│  │ • Feasibility flag (0 or 1)                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ACTION (what the agent does):                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • 7 continuous values [-1, 1] → new design params    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  REWARD (feedback to agent):                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ reward = -cost                    (minimize cost)    │   │
│  │        - penalty * violations     (stay in bounds)   │   │
│  │        + bonus if feasible        (encourage valid)  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component 5: Neural Optimizer (`ml/inference/neural_optimizer.py`)

### What It Does
The **unified API** that combines everything for actual use.

### Three Optimization Modes

```
MODE 1: optimize_fast() - Pure Neural (fastest)
─────────────────────────────────────────────────
1. Generate 1000 random candidates
2. (Optional) Add candidates from inverse model
3. Evaluate ALL with surrogate model (~1ms total)
4. Pick best, refine with gradient descent
5. Return result

Speed: ~50-100ms
Quality: Good (depends on surrogate accuracy)


MODE 2: optimize_hybrid() - Neural + Physics (balanced)
─────────────────────────────────────────────────────────
1. Run optimize_fast() 100 times
2. Get top 10 candidates
3. Verify each with REAL physics calculations
4. Return best verified result

Speed: ~5-10 seconds
Quality: Excellent (physics-verified)


MODE 3: inverse_design() - Specs to Design
──────────────────────────────────────────────
1. User provides: "I need NLL<800W, LL<7000W, Ucc=6%"
2. CVAE generates multiple design options
3. Return list of valid designs

Speed: ~10ms
Use case: "What designs can achieve these specs?"
```

---

## How Everything Connects

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                               │
│                                                                      │
│   mainRect.py ─────────────────────┐                                │
│   (Your physics)                   │                                │
│         │                          ▼                                │
│         │               ┌───────────────────┐                       │
│         └──────────────▶│ generate_dataset  │                       │
│                         │ (100k samples)    │                       │
│                         └─────────┬─────────┘                       │
│                                   │                                 │
│                    ┌──────────────┼──────────────┐                  │
│                    ▼              ▼              ▼                  │
│              ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│              │Train     │  │Train     │  │Train     │              │
│              │Surrogate │  │Inverse   │  │RL Agent  │              │
│              └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│                   │             │             │                     │
│                   ▼             ▼             ▼                     │
│              ┌──────────────────────────────────────┐              │
│              │        Saved Model Checkpoints       │              │
│              └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                               │
│                                                                      │
│   User Request: "Optimize for 1000kVA transformer"                  │
│         │                                                            │
│         ▼                                                            │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │              NeuralOptimizer                               │     │
│   │  ┌─────────────┬─────────────┬─────────────┐              │     │
│   │  │ Surrogate   │  Inverse    │  RL Policy  │              │     │
│   │  │ Model       │  Model      │  (optional) │              │     │
│   │  └─────────────┴─────────────┴─────────────┘              │     │
│   │                        │                                   │     │
│   │                        ▼                                   │     │
│   │  ┌───────────────────────────────────────────┐            │     │
│   │  │ optimize_fast() / optimize_hybrid() /     │            │     │
│   │  │ inverse_design()                          │            │     │
│   │  └───────────────────────────────────────────┘            │     │
│   └───────────────────────────────────────────────────────────┘     │
│         │                                                            │
│         ▼                                                            │
│   Optimal Design: core_dia=185mm, lv_turns=28, ... @ $8,500         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Commands

```bash
# Step 1: Generate training data (run once, takes ~30 min)
cd TrafoDes
python -m ml.data.generate_dataset --n_samples 100000

# Step 2: Train the surrogate model (run once, takes ~10-30 min)
python -m ml.training.train_surrogate --epochs 100

# Step 3: Use the trained model
python << 'EOF'
from ml.inference.neural_optimizer import NeuralOptimizer

# Load trained models
opt = NeuralOptimizer()
opt.load_models('ml/checkpoints')

# Fast optimization
params, cost = opt.optimize_fast()
print(f"Optimal design cost: ${cost:.2f}")
print(f"Parameters: {params}")

# Or inverse design
designs = opt.inverse_design(target_nll=800, target_ll=7000, target_ucc=6.0)
print(f"Found {len(designs)} valid designs")
EOF
```

---

## Key Benefits

| Aspect | Physics-Only | With ML |
|--------|--------------|---------|
| Single evaluation | 5-60 sec | <1 ms |
| Full optimization | 30-300 sec | 50-100 ms |
| Inverse design | Not possible | Instant |
| Batch processing | Sequential | Parallel on GPU |

---

## Files Created

```
ml/
├── __init__.py                    # Package init
├── data/
│   ├── __init__.py
│   └── generate_dataset.py        # Training data generation
├── models/
│   ├── __init__.py
│   ├── surrogate.py               # Fast prediction model
│   └── inverse.py                 # Specs → Params model
├── envs/
│   ├── __init__.py
│   └── transformer_env.py         # RL training environment
├── training/
│   ├── __init__.py
│   └── train_surrogate.py         # Training script
└── inference/
    ├── __init__.py
    └── neural_optimizer.py        # Unified API
```
