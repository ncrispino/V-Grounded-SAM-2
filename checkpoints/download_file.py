import os
import requests

# Base URL for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
checkpoints = {
    "sam2.1_hiera_tiny.pt": f"{SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt": f"{SAM2p1_BASE_URL}/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": f"{SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt": f"{SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"
}

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
for filename, url in checkpoints.items():
    download_file(url, filename)

print("All checkpoints are downloaded successfully.")
