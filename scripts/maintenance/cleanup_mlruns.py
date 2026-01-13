import os
import shutil
import sys
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
print(f"Cleaning up {MLRUNS_DIR}...")

if not os.path.exists(MLRUNS_DIR):
    print("mlruns directory not found.")
    sys.exit(1)

experiments = []
for d in os.listdir(MLRUNS_DIR):
    full_path = os.path.join(MLRUNS_DIR, d)
    if os.path.isdir(full_path) and d.isdigit():
        experiments.append((full_path, os.path.getmtime(full_path)))

if not experiments:
    print("No experiments found.")
    sys.exit(0)

# Sort by mtime descending (newest first)
experiments.sort(key=lambda x: x[1], reverse=True)

latest_exp = experiments[0]
latest_time = datetime.datetime.fromtimestamp(latest_exp[1]).strftime('%Y-%m-%d %H:%M:%S')
print(f"Keeping latest experiment: {latest_exp[0]} (Modified: {latest_time})")

for exp_path, mtime in experiments[1:]:
    time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Deleting experiment: {exp_path} (Modified: {time_str})")
    try:
        shutil.rmtree(exp_path)
    except Exception as e:
        print(f"Error deleting {exp_path}: {e}")

print("Cleanup complete.")
