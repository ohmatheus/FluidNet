# ML Training Pipeline

This is where the neural network learns to be a fluid simulator. I train a UNet on sequences from Blender/Mantaflow simulations, then export it to ONNX so the C++ engine can run it in real-time.

## Training Data

The training data comes from the [vdb-tools](../vdb-tools/README.md) pipeline - 3D Blender simulations projected onto 2D and saved as NPZ files. Each sequence contains a few hundred frames of fluid behavior: smoke rising, obstacles blocking flow, emitters spawning density.

I tried to create diverse training scenarios:
- Simple setups (single emitter, empty space)
- Complex shapes (multiple emitters, obstacles everywhere)
- Empty scenes (no emitter active, just obstacles and initial velocity)
- Roughly 50/50 split on colliders vs no colliders

The idea was to teach the network a variety of scenarios so it generalizes. Though honestly, with more experience, the data generation could probably use some improvements.

## The Model

A UNet - encoder-decoder architecture with skip connections. Nothing fancy, just a solid convolutional network.

**Input:** Current state (density + velocity) + previous frame density + emitter/collider masks (6 channels total)

**Output:** Next frame (density + velocity, 3 channels)

I experimented a lot with the architecture:
- Different depths (how many downsampling/upsampling stages)
- Base channel counts (how wide the network is)
- Normalization layers (batch norm, instance norm, group norm, or none)
- Activation functions (ReLU, LeakyReLU, GELU, SiLU)
- Upsampling methods (bilinear interpolation vs learned transpose convolutions)

Multiple model sizes were tested (small, medium) trying to balance quality vs inference speed. Smaller models run faster but struggle with complex physics like collider interactions.

## Multistep K Training

This is the game-changer that made everything work more realistically.

**The Problem:** When you train a model to predict just one frame at a time (single-step, K=1), it works fine during training. But when you run it autoregressively - feeding its own predictions back as input for hundreds of frames - it drifts. Errors accumulate and the simulation becomes unstable.

**The Solution:** Multistep training (K=2, K=3, K=4).

Instead of training the model to predict frame `t+1` from frame `t`, I train it to predict K frames into the future *during training*. The model outputs frame `t+1`, then that prediction becomes the input for predicting `t+2`, and so on for K steps.

This forces the network to learn dynamics that are stable over longer horizons. It has to deal with its own errors during training, so it learns to not make them in the first place.

**Why it matters:**
- Without K training: models drift after a few hundreds frames
- With K training: stable rollouts for hundreds of frames
- This is what enables the real-time engine to run autoregressively without exploding

I started with K=1, then moved to K=2, then K=3, eventually K=4. The higher K, the more stable the long-term rollouts. Training takes longer (you're computing more forward passes per batch), but the improvement in rollout quality is massive.

**Weight decay experiments:** I also experimented with rollout weight decay - how much to weight each step in the K-step loss. Tried 1.0 (equal weighting), 0.80, 1.10, 1.50:
- Values < 1.0 smoothed the simulation
- Too high made it unstable
- Slightly > 1.0 (like 1.10) gave more emphasis on later frames - this worked best for me

**Gradient truncation:** I kept gradients flowing through the rollout (no detach between steps) which helped fluid quality, though it uses more memory. The current implementation isn't optimal, but it's flexible enough to experiment further - in theory could try K=10 or K=20, though not sure my GPU could handle it.

The training system allows configuring parameters independently for each K value, making it easy to experiment with different configurations (loss weights, learning rate, etc.) without starting from scratch.

Single-step training was the naive approach. Multistep autoregressive training is what makes a neural fluid simulator actually work in practice.

### Rollout Weight Decay: How It Works

The `rollout_weight_decay` parameter (currently 1.10) controls how much each step in the K-rollout contributes to the total loss. Values > 1.0 progressively emphasize later frames, which is important for long-term stability.

**For K=3 with weight_decay=1.10, the weights are [1.0, 1.10, 1.21]:**

The total gradient combines direct and indirect paths:
```
∂total_loss/∂θ includes:

Direct gradients (each step's loss computed independently):
  w₀/sum × ∂loss₀/∂θ                     [direct from step 0]
+ w₁/sum × ∂loss₁/∂θ                     [direct from step 1]
+ w₂/sum × ∂loss₂/∂θ                     [direct from step 2]

Indirect gradients (flowing back through previous predictions):
+ w₁/sum × ∂loss₁/∂pred₀ × ∂pred₀/∂θ     [step 1 through step 0]
+ w₂/sum × ∂loss₂/∂pred₁ × ∂pred₁/∂θ     [step 2 through step 1]
+ w₂/sum × ∂loss₂/∂pred₀ × ∂pred₀/∂θ     [step 2 through step 0]

Where sum = w₀ + w₁ + w₂ (normalization)
```

**Why values > 1.0 help:**
- Emphasizes later frames in the rollout
- Model learns to prioritize long-term accuracy over immediate prediction
- Helps combat error accumulation in autoregressive inference
- Too high (>1.5) causes instability; 1.10 is a sweet spot

Since gradient truncation is disabled (`rollout_gradient_truncation: false`), all these indirect gradient paths remain active. This means the network sees how its current-step predictions affect future steps during training.

## Loss Functions & Experiments

I started out with a bunch of physics-aware losses:
- **MSE** (mean squared error) - the baseline reconstruction loss
- **Divergence penalty** - encourages velocity fields to be incompressible
- **Emitter loss** - prevents the model from hallucinating density in empty regions
- **Gradient loss** - helps preserve sharp smoke boundaries

Initially these felt important. But once I switched to multistep K training, most of them became less critical. The K rollout itself naturally forces better physical behavior - if the model violates physics, the rollout explodes during training and the loss goes up.

I'm still using divergence penalty, but honestly unsure if it's actually helping. Mostly experimentation at this point to figure out what matters.

## Curriculum Learning in Practice

The K1→K2→K3→K4 progression is built into the training system through a hierarchical variant configuration.

### The Variant Hierarchy

Each variant file specifies its rollout step and parent:

```yaml
# K1: Start from scratch
variants/experiments/only_div/001/K1-Unet_medium-only_div_001.yaml
  rollout_step: 1
  parent_variant: null
  learning_rate: 0.001
  epochs: 3

# K2: Load K1 weights, train 2-step rollout
K2-Unet_medium-only_div_001.yaml
  rollout_step: 2
  parent_variant: K1-Unet_medium-only_div_001
  learning_rate: 0.0006
  epochs: 6

# K3: Load K2 weights, train 3-step rollout
K3-Unet_medium-only_div_001.yaml
  rollout_step: 3
  parent_variant: K2-Unet_medium-only_div_001
  learning_rate: 0.0003
  epochs: 9

# K4: Load K3 weights, train 4-step rollout (with plateau LR scheduler and early stopping)
K4-Unet_medium-only_div_001.yaml
  rollout_step: 4
  parent_variant: K3-Unet_medium-only_div_001
  learning_rate: 0.0001
  epochs: 100
```

### Why Learning Rate Decreases

Each K stage is fine-tuning on top of already-learned physics:
- **K1:** Learning basic fluid dynamics from scratch → higher LR okay
- **K2-K4:** Extending temporal horizon on stable features → lower LR prevents disrupting what's already learned

The progressive LR reduction matches the increasing task difficulty and decreasing "room for improvement."

### Automatic Parent Loading

The `variant_manager.py` system handles checkpoint inheritance automatically. When you train K4, it:
1. Resolves the dependency chain: K4 → K3 → K2 → K1
2. Loads `best_model.pth` from K3's checkpoint directory
3. Initializes K4 model with K3 weights
4. Starts training with K4's config (rollout_step=4, lower LR)

This means each stage starts from a model that already understands fluid physics at the (K-1) level.

### Systematic Experimentation

The variant system enables organized ablation studies:
- **baseline/**: Full physics loss (MSE + divergence + emitter + gradient)
- **no_phys/**: MSE only, no physics constraints
- **only_div/001/**: MSE + 0.001 divergence weight
- **only_div/005/**: MSE + 0.01 divergence weight

Each experiment family has its own K1→K4 progression. This makes it easy to compare: "Does higher divergence weight improve K4 stability?" without re-implementing everything.

## Validation

Physics metrics are tracked during training:
- Per-channel MSE (density, velocity)
- Divergence norm (how incompressible the flow is)
- Kinetic energy (to detect instability)
- Collider violation (density inside solid obstacles)

<p align="center">
  <img src="../assets/val_metrics.png" alt="Validation metrics over training" width="700"/>
</p>
<p align="center"><em>Validation metrics tracked during training</em></p>

But those metrics don't tell the whole story. A model can have low divergence and still produce unconvincing fluid motion. It was important for me to validate in real conditions: export the model to ONNX, load it in the [C++ engine](../engine/README.md), and interact with it. Does the fluid swirl naturally? Do colliders work? Can you inject forces without the simulation breaking?

That qualitative evaluation is where you actually learn what the model can and can't do.


### Alternative Approaches Tried

**Gradient loss (sharp features):**
- Preserved sharp smoke boundaries during training
- But created artifacts during long autoregressive rollouts
- Trade-off: single-frame sharpness vs 600-frame stability → chose stability

**Emitter loss (prevent hallucinations):**
- Successfully prevented density spawning in empty regions
- But caused instability in some scenarios
- Interfered with natural smoke advection

**Full physics loss (MSE + div + emitter + gradient):**
- Too many competing objectives
- Model couldn't balance them effectively
- Worse results than divergence-only

**Current best: MSE + divergence only**
- Simpler loss landscape
- Model focuses on core physics (incompressibility)
- K-rollout naturally enforces other constraints through temporal consistency

## The Workflow

**1. Train**
```bash
cd ml
python scripts/train.py variants K3-Unet_medium-baseline
```

Training runs for a couple hundred epochs. MLflow tracks metrics. Checkpoints save automatically. Early stopping kicks in if validation loss plateaus.

**2. Export to ONNX**
```bash
python scripts/export_to_onnx.py
```

This converts the PyTorch model to ONNX format. The engine needs ONNX for inference.

**3. Test in Engine**

Load the exported model in the [C++ engine](../engine/README.md) and see how it performs in real-time. Try different scenarios. Push it until it breaks.

**4. Iterate**

Based on how the engine rollout looks, adjust training (more K steps? different loss weights? better data?) and repeat.

## Getting Started

**Train a model:**
```bash
cd ml
python scripts/train.py variants K3-Unet_medium-baseline
```

Check `config/variants/` for different experiment configurations. Variants use hierarchical inheritance so you can quickly try modifications (different K values, loss weights, architecture tweaks) without duplicating config.

**Monitor training:**
```bash
mlflow ui
```

Then navigate to `http://localhost:5000` to see training curves and metrics.

**Export for deployment:**
```bash
python scripts/export_to_onnx.py
```

The ONNX model goes to `data/onnx/` where the engine can load it.

## Limitations & Future Work

The current model achieves stable 600+ frame rollouts, which was the goal. But there's definitely room for improvement.

### Current Limitations

**Divergence norm higher than ideal:**
- Currently stabilizes around 0.30
- Ideal would be closer to 0.05-0.10 for stricter incompressibility
- Affects long-term mass conservation slightly
- Trade-off: lower divergence often creates other artifacts

**Validation plateau while Train don't:**
- Suggests limited dataset diversity
- Model exhausts learnable signal early
- More varied scenarios would likely push this later

**Training data bottleneck:**
- Limited by single-machine Blender simulation generation
- Creating and processing sims takes hours
- Dataset size: enough for proof of concept, but more data = better generalization
- Need: more emitter/collider configurations, wider range of initial conditions, turbulent scenarios

**Architecture exploration limited:**
- Only tried UNet variants (small, medium, large)
- Didn't experiment with:
  - Fourier Neural Operators (FNO) - better at PDEs theoretically
  - Transformers for temporal attention
  - Hybrid CNN + attention architectures
  - Graph neural networks for irregular domains

### What Could Be Better

**Physics accuracy vs stability:**
- Current model prioritizes rollout stability over single-frame physics accuracy
- A better model would achieve both
- Might require more sophisticated loss balancing or architecture

**Small models & colliders:**
- Small UNet (fast inference) struggles with collider interactions
- Medium UNet handles colliders well but slower
- Goal: match medium's physics quality at small's speed

**Quantization:**
- INT8 quantization works but slightly degrades collider behavior
- Need better calibration or quantization-aware training

### What I'd Try Next

1. **More training data:**
   - Generate 2-3x more Blender simulations
   - Better scenario coverage (complex multi-collider setups, high-velocity turbulence)
   - Expected result: validation plateau pushed further, better generalization

2. **Architecture experiments:**
   - FNO: theoretically better for PDEs, might achieve lower divergence
   - Vision transformers: could capture long-range fluid interactions better
   - Hybrid: CNN encoder + transformer temporal reasoning

3. **Loss function refinement:**
   - Adaptive loss weighting (automatically balance MSE vs divergence)
   - Multi-scale divergence penalty (enforce incompressibility at different resolutions)
   - Adversarial training (like tempoGAN) for more realistic fluid textures

4. **Better validation:**
   - Automated rollout tests during training (not just single-step validation)
   - Physics-based metrics at multiple K values (K=10, K=20, K=50 rollout tests)
   - Benchmark against traditional solvers on standardized scenarios

The current model works and achieves the project goals. But these improvements would take it from "works well enough" to "approaches production quality."

This pipeline produces models that run in real-time and simulate believable fluid dynamics, now it's about refinement.
