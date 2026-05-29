# src/dataset_writer.py
import json
from pathlib import Path


def save_jsonl(record: dict, output_path: str):
    """
    하나의 caption 데이터를 JSONL 파일에 append한다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_jsonl_if_valid(record: dict, output_path: str):
    """
    caption 결과가 valid이고 quality_score가 충분히 높을 때만 JSONL에 저장한다.
    """
    if not record.get("is_valid", False):
        print("Skipped: invalid caption result.")
        return

    if record.get("quality_score", 0) < 0.8:
        print("Skipped: quality score is too low.")
        return

    save_jsonl(record, output_path)
    print(f"Saved valid record to {output_path}")