import matplotlib.pyplot as plt
import csv
import os

# Read the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'hw1/exp/flow/log.csv')
steps = []
losses = []

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Skip rows with empty loss values
        if row['train/loss'] and row['train/loss'].strip():
            try:
                step = int(row['step'])
                loss = float(row['train/loss'])
                steps.append(step)
                losses.append(loss)
            except (ValueError, KeyError):
                continue

print(f"Number of loss data points: {len(steps)}")

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, linewidth=1.5)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Loss Curve (Flow)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_path = os.path.join(script_dir, 'hw1/loss_curve_flow.png')
plt.savefig(output_path, dpi=150)
print(f"Plot saved to {output_path}")

plt.show()
