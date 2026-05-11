# Reproduction Guide

This document provides step-by-step instructions for reproducing the exact results from the paper.

## Environment Setup

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository and set up dependencies

```bash
git clone https://github.com/NPLawrence/soapy-rl.git
cd soapy-rl
uv sync
```

The `uv.lock` file ensures reproducible dependency versions across all installations.

## Reproducing Exact Paper Results

### Option A: Full Multi-Run Sweep (Recommended)

The paper results come from a sweep of 20 experiments: **10 random seeds × 2 optimizers** (SOAP vs AdamW).

```bash
source .venv/bin/activate
python scripts/training/main_trainer.py
```

**Expected runtime**: ~4–8 hours depending on hardware (GPU highly recommended)

**Output**: 
- 20 trained models in `runs/` directories
- TensorBoard logs for each run
- Automatic W&B logging (if enabled)

### Option B: Quick Validation (Single Run)

To verify the setup works without running the full sweep:

```bash
source .venv/bin/activate
python train.py --seed=1 --optimizer=soap --total_epochs=10
```

This runs one epoch of SOAP training to verify all dependencies and GPU access work correctly.

## Full Sweep Details

### Sweep Configuration (`scripts/training/main_trainer.py`)

The multi-run sweep launches experiments with:

- **Optimizers**: SOAP, AdamW
- **Seeds**: 1–10 (ensures statistical significance)
- **Environment**: Inverted Pendulum (`pendulum`)
- **Duration**: 200 epochs per run
- **Batch size**: 512
- **Learning rate**: 1e-4

### Command-Line Override Example

```bash
# Run single experiment with custom settings
python train.py \
  --seed=42 \
  --optimizer=soap \
  --learning_rate=2e-4 \
  --total_epochs=500 \
  --cuda=True
```

## Downloading and Analyzing Results

### 1. Sync results with Weights & Biases

```bash
source .venv/bin/activate
python scripts/analysis/wandb_data.py
```

This downloads run histories as local pickle files in a `wandb_data/` directory.

### 2. Generate comparison plots

```bash
python scripts/analysis/plot.py
```

Output plots will be saved to `figures/` directory.

## Interpreting Results

### Output Directories

Each training run creates a timestamped directory in `runs/`:

```
runs/
└── pendulum__train__1__1763403959/
    ├── events.out.tfevents.1763403959.hostname.profile  # TensorBoard data
    ├── train.cleanrl_model                              # Saved model weights
    └── checkpoints/                                      # (if enabled)
```

### TensorBoard Visualization

```bash
tensorboard --logdir=runs/
```

Open [http://localhost:6006](http://localhost:6006) in your browser to view training curves.

### Key Metrics to Monitor

1. **`losses/q_loss`**: Main critic/Q-function loss
2. **`losses/eval_return`**: Evaluation trajectory loss
3. **`losses/eval_error`**: Q-value prediction error on evaluation set
4. **`plots/dip_plot`**: Visualization of pendulum trajectory over time

### Expected Performance

Paper results show:
- **SOAP**: Faster convergence and better final performance on pendulum task
- **AdamW**: Baseline for comparison, typically slower convergence

Exact numbers depend on random seed and hardware (especially GPU acceleration).

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
python train.py --batch_size=256
```

Or reduce dataset size:
```bash
python train.py --num_data=50000
```

### CUDA Not Found

Ensure PyTorch with CUDA is installed. `uv sync` should handle this, but to verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Weights & Biases Authentication Issues

```bash
wandb login
```

Or disable W&B logging:
```bash
python train.py --track=False
```

## Hardware Requirements

### Minimum
- Python >=3.10
- 8GB RAM
- CPU: ~4 cores (>2x slower than GPU)

### Recommended (for fastest results)
- GPU: NVIDIA GPU with CUDA support (A100, RTX 3090, etc.)
- RAM: 16+ GB
- CPU: 8+ cores for data loading

### Estimated Runtime

| Configuration | Single Run | Full Sweep (20 runs) |
|---|---|---|
| GPU (A100/RTX3090) | 15–30 min | 5–10 hours |
| GPU (RTX 2080 Ti) | 30–60 min | 10–20 hours |
| CPU (8 cores) | 2–4 hours | 40–80 hours |

## Verifying Reproducibility

To verify your environment matches the paper:

```bash
# Check Python version
python --version  # Should be >=3.10

# Check PyTorch version
python -c "import torch; print(torch.__version__)"  # Should match uv.lock

# Verify uv.lock is being used
uv pip freeze | head -10
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{...
  title={...},
  author={...},
  year={2025}
}
```

## Questions?

For issues or questions about reproduction:
1. Check the [main README.md](README.md) for general setup
2. Review [train.py](train.py) for all available CLI arguments
3. Open an issue on GitHub with details about the error

---

**Last updated**: December 2024
