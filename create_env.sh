pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install requests

cd checkpoints
python download_file.py

cd ../gdino_checkpoints
python download_file.py

cd ..
pip install -e .
pip install --no-build-isolation -e grounding_dino

pip install opencv-python-headless==4.5.5.64  # or just opencv-python.
pip install supervision-0.16.0

pip install transformers
