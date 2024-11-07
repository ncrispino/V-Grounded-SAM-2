import os
import requests

# Base URL for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# Define the URLs for the checkpoints
BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/"
swint_ogc_url=f"{BASE_URL}v0.1.0-alpha/groundingdino_swint_ogc.pth"
swinb_cogcoor_url=f"{BASE_URL}v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"

def download_file(url, dest_path):
    """Downloads a file given a URL and destination path."""
    try:
        print(f"Downloading from {url} to {dest_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
        print(f"Downloaded {dest_path} successfully.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# Download each checkpoint
download_file(swint_ogc_url, "groundingdino_swint_ogc.pth")
download_file(swinb_cogcoor_url, "groundingdino_swinb_cogcoor.pth")

print("All checkpoints are downloaded successfully.")
