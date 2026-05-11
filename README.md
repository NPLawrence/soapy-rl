# soapy-rl

Codebase for offline DQN-style control experiments with **SOAP** and **AdamW** optimizers on pendulum dynamics. This repository implements the SOAP optimizer from [SOAP: Improving and stabilizing shampoo using adam for language modeling](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e988664070e9591f93fdcf605f7dc623-Abstract-Conference.html) for neural fitted Q-iteration tasks.


- **SOAP Optimizer**: Approximate Gauss-Newton training
- **Training Framework**: Discrete-action control and Q-iteration training using Neuromancer and PyTorch
- **Dynamics**: Inverted Pendulum and Acrobot systems modeled as ODEs
- **Reproducibility**: Experiment sweeps with multiple seeds and W&B integration for tracking results

## Repository Structure

```
soapy-rl/
├── soap.py                      # SOAP optimizer implementation (https://arxiv.org/abs/2409.11321)
├── train.py                     # Main training entry point for single runs
├── pendulum.py                  # ODE dynamics: InvertedPendulum, Acrobot, DoubleInvertedPendulum, DoubleIntegrator
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                      # Reproducible dependency lock file (uv package manager)
├── scripts/
│   ├── training/
│   │   └── main_trainer.py      # Multi-run sweep launcher (20 experiments: 10 seeds × 2 optimizers)
│   └── analysis/
│       ├── wandb_data.py        # Download W&B run histories to local pickle files
│       └── plot.py              # Generate comparison plots from processed run data
├── archive/                     # Historical experiments (reference implementations)
├── data/
│   ├── raw/                     # Seed data and raw inputs
│   └── processed/               # Generated processed datasets (.pkl files)
├── figures/                     # Generated plot outputs
├── runs/                        # TensorBoard logs and checkpoints (not committed)
└── wandb/                       # W&B cache (not committed)
```

## Quick Start

Installation is reproducible via `uv.lock`.

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync dependencies and create virtual environment**:
   ```bash
   uv sync
   ```

3. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Run training**:
   ```bash
   python train.py
   ```

## Running Experiments

### Single Training Run
```bash
# Default configuration (SOAP optimizer, 200 epochs, seed=1)
python train.py

# With custom parameters
python train.py --seed=42 --optimizer=adamw --learning_rate=0.001 --total_epochs=100
```

### Multi-Run Sweep (Reproducing Paper Results)
```bash
# Runs 20 experiments: 10 seeds × 2 optimizers (SOAP vs AdamW)
python scripts/training/main_trainer.py
```

### Retrieve Results from Weights & Biases
```bash
# Download run histories as local pickle files
python scripts/analysis/wandb_data.py
```

### Generate Comparison Plots
```bash
# Create plots comparing optimizer performance
python scripts/analysis/plot.py
```

## Configuration

### Command-Line Arguments (train.py)

- `--seed`: Random seed (default: 1)
- `--optimizer`: Optimizer choice: `soap`, `adam`, `adamw`, `sgd`, `rmsprop`, `asgd` (default: `soap`)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--total_epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size for training (default: 512)
- `--num_data`: Size of initial dataset (default: 100000)
- `--track`: Enable Weights & Biases logging (default: True)
- `--wandb_project_name`: W&B project name (default: `cleanRL`)
- `--save_model`: Save trained model to disk (default: True)
- `--env_id`: Environment name used in run naming (default: `pendulum`)
- `--cuda`: Use GPU if available (default: True)

## Weights & Biases Integration

Experiment tracking is enabled by default and logged to W&B. To use this feature:

1. Create a [Weights & Biases](https://wandb.ai/) account
2. Authenticate locally:
   ```bash
   wandb login
   ```
3. Runs will automatically sync to your W&B project (`cleanRL` by default)

To disable W&B logging:
```bash
python train.py --track=False
```

## Output Structure

- **`runs/{run_name}/`**: TensorBoard logs and model checkpoints
  - `runs/{run_name}/{exp_name}.cleanrl_model`: Saved neural network weights
  - `events.out.tfevents.*`: TensorBoard event files (training curves, metrics)
- **`figures/`**: Generated comparison plots (matplotlib/plotly)
- **`data/processed/`**: Processed datasets (pickle files)
- **`wandb/`**: Local W&B cache (not committed to version control)

## Dependencies

Key dependencies (see `pyproject.toml` for full list):

- **PyTorch**: `torch>=2.6.0` — Deep learning framework
- **Neuromancer**: `neuromancer>=1.5.6` — Physics-informed ML and systems modeling
- **Gymnasium**: `gymnasium>=1.2.2` — RL environment API
- **W&B**: `wandb>=0.21.0` — Experiment tracking
- **TensorBoard**: `tensorboard>=2.20.0` — Visualization
- **tyro**: `tyro>=0.9.35` — CLI argument parsing

## Notes

- **Reproducibility**: The `uv.lock` file ensures exact reproducibility of dependencies. Always use `uv sync` to maintain consistency.
- **Archive**: The `archive/` directory contains historical experiments and alternative implementations. These are included for reference but are not part of the primary release.
- **Development Environment**: Active TensorBoard logs and W&B caches are stored locally in `runs/` and `wandb/` but are not committed to version control.
- **GPU**: GPU acceleration is enabled by default if CUDA is available. To force CPU, use `--cuda=False`.

## Paper Reference

If you use this code, please cite our paper:

```bibtex
@article{bestmckay2026errorwhitening,
  title={Error whitening: Why {Gauss-Newton} outperforms {Newton}},
  author={Best Mckay, Maricela and Lawrence, Nathan P and Wetton, Brian and Gopaluni, Bhushan},
  year={2026}
}
```
