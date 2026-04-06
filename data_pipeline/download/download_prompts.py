import logging
import json
from tqdm import tqdm
from pathlib import Path

from datasets import load_dataset

def download_and_process_h4_prompts(limit: int = 0):
    """
    limit 如果是 0 那么 下载全部数据集，不然就下载limit条数据
    """
    if not load_dataset:
        return

    repo_id = "HuggingFaceH4/instruction-dataset"

    subset = "test"

    output_dir = Path(__file__).parent.parent / "prompts"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "h4_prompts.txt"
    jsonl_cache_path = output_dir / "h4_test.jsonl"

    if output_path.exists():
        logging.info(f"Loading existing prompts file: {output_path}")
        return

    logging.info(f"Downloading prompts from {repo_id}")

    try:
        if limit == 0:
            dataset = load_dataset(repo_id, split=f"{subset}", streaming=False)
        else:
            dataset = load_dataset(repo_id, split=f"{subset}[:{limit}]", streaming=False)

        dataset.to_json(jsonl_cache_path)
        logging.info(f"cache prompts to {jsonl_cache_path}")
        logging.info(f"processing jsonl file to {output_path}")

        with open(jsonl_cache_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
            for line in tqdm(f_in, desc="Extracting prompts"):
                try:
                    data = json.loads(line)
                    prompt = data.get("prompt")
                    if prompt:
                        f_out.write(prompt.strip() + "\n")

                except Exception as e:
                    logging.warning(f"json line error: {line}")
        logging.info(f"cache prompts to {output_path}")
    except Exception as e:
        logging.error(f"Exception while downloading prompts: {e}")

if __name__ == "__main__":
    download_and_process_h4_prompts(0)


