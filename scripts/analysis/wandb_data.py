import numpy as np
import pandas as pd
import wandb
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "data" / "processed" / "all_data.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    # Update project name to match your W&B project from main_trainer.py
    project_name = "nplawrence/soapy-rl-experiments"
    runs = api.runs(project_name)

    run_name_list = []
    optimizer_list = []
    learning_rate_list = []
    num_data_list = []
    eval_list = []
    train_td_error_list = []
    eval_td_error_list = []
    dataset = {"run_name": run_name_list, "optimizer": optimizer_list, "num_data": num_data_list,
               "learning_rate": learning_rate_list, "eval": eval_list, "train_td_error": train_td_error_list,
               "eval_td_error": eval_td_error_list}
    for (i, run) in enumerate(runs):
        config = {k: v for k,v in run.config.items()
              if not k.startswith('_')}

        run_name_list.append(run.name)
        optimizer_list.append(config['optimizer'])
        learning_rate_list.append(config['learning_rate'])
        num_data_list.append(config['num_data'])

        history = run.scan_history()
        eval_performance = [row["losses/eval_return"] for row in history if row["losses/eval_return"] is not None]
        eval_error = [row["losses/eval_error"] for row in history if row["losses/eval_error"] is not None]
        train_error = [row["losses/q_loss"] for row in history if row["losses/q_loss"] is not None]

        eval_list.append(np.array(eval_performance))
        eval_td_error_list.append(np.array(eval_error))
        train_td_error_list.append(np.array(train_error))

        print(i)

    runs_df = pd.DataFrame(dataset)
    runs_df.to_pickle(output_path)


if __name__ == "__main__":
    main()