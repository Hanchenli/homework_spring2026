from pathlib import Path
from wandb.apis.public.api import Api

# Initialize API
api = Api()

# Get run object
run = api.run("lihanc-university-of-california-berkeley/hw1-imitation/runs/61ve7ex2")

# Create download directory
download_dir = Path("download_directory")
download_dir.mkdir(exist_ok=True)

# Iterate over files in the run
for file in run.files():
    print(f"Downloading: {file.name}")
    print(f"  URL: {file.url}")
    print(f"  Size: {file.size} bytes")
    
    # Download the file
    file.download(root=str(download_dir), replace=True)
    print(f"  ✓ Downloaded to {download_dir / file.name}\n")

print(f"All files downloaded to {download_dir}")