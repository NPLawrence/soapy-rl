"""Run a bunch of experiments in a loop."""

from pathlib import Path
import runpy
import sys

project_name = "soapy-rl-experiments"  # Change to your W&B project name
trials = 10

optimizer = ['soap', 'adamw']
num_data = [40000]
# critic_loss = ['mse']
batch_size = [512]
learning_rate = [1e-4]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    trainer = repo_root / "train.py"

    experiment_num = 0
    for seed in range(trials):
        for opt in optimizer:
            for nd in num_data:
                for bs in batch_size:
                    if opt == "adamw":
                        lr = 1e-5
                    else:
                        lr = 1e-4

                    experiment_num += 1

                    print("Experiment: ", experiment_num)
                    sys.argv = [
                        "",
                        f"--seed={int(seed)}",
                        f"--batch_size={int(bs)}",
                        f"--optimizer={str(opt)}",
                        f"--num_data={int(nd)}",
                        f"--learning_rate={lr}",
                        f"--wandb_project_name={project_name}",
                    ]

                    runpy.run_path(path_name=str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()

