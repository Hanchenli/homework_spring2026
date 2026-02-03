import matplotlib.pyplot as plt
import pandas as pd
from wandb.apis.public.api import Api

# Initialize API
api = Api()

# Get run object - the run with seed_42_20260202_004718
run = api.run("lihanc-university-of-california-berkeley/hw1-imitation/runs/61ve7ex2")

print(f"Run name: {run.name}")
print(f"Run ID: {run.id}")

# Get run history
history = run.history()
print(f"History columns: {history.columns.tolist()}")

# Filter to get loss data
loss_data = history[['_step', 'train/loss']].dropna()
print(f"Number of loss data points: {len(loss_data)}")

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(loss_data['_step'], loss_data['train/loss'], linewidth=1.5)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title(f'Loss Curve for {run.name}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_path = 'loss_curve_seed_42.png'
plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")

plt.show()
