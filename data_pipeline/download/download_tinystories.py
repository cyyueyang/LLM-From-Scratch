import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_tinystories():
    dataset_name = "roneneldan/TinyStories"
    save_path = Path(__file__).parent.parent / "downloaded_data" / "TinyStories"

    if os.path.exists(save_path):
        print("already downloaded")
        return

    print(f"downloading tinystories from {dataset_name}")

    save_path.mkdir(parents=True, exist_ok=True)
    full_dataset = load_dataset(dataset_name, split="train")
    full_dataset.save_to_disk(str(save_path))

    print(f"saved tinystories to {save_path}")

if __name__ == "__main__":
    download_tinystories()

