import wandb
import numpy as np

ENTITY = 'fomo_aiisdh'
PROJECT = 'BiLaMa'
METRIC_NAME = 'test/avg_psnr'

api = wandb.Api(timeout=19)

runs = api.runs(f"{ENTITY}/{PROJECT}")

for i, run in enumerate(runs):

    df = run.history()
    if df.empty or METRIC_NAME not in df.columns:
        print(f"Run {i} has no history or no {METRIC_NAME} metric")
        continue
    values = df[METRIC_NAME].values
    values = values[~np.isnan(values)]

    run.summary[f"{METRIC_NAME}_max"] = np.max(values)
    # run.summary[f"{METRIC_NAME}_min"] = np.min(values)
    # run.summary[f"{METRIC_NAME}_std"] = np.std(values)
    print(f"{i+1}/{len(runs)} {run.name:<48} {np.max(values):.02f}")

    run.summary.update()
exit()