"""
extract_sd35_prompt_caption.py

sd35_generation_dataset_with_spec.jsonl에서 이미지 파일 번호 기준으로
SD3.5용 prompt와 visual/display caption을 각각 텍스트 파일로 추출한다.

입력:
    data/output/sd35_generation_dataset_with_spec.jsonl

출력:
    data/output/prompt_sd35.txt
    data/output/caption_sd35.txt

실행:
    python extract_sd35_prompt_caption.py
"""

import json
from pathlib import Path


INPUT_PATH = Path("data/output/sd35_generation_dataset_with_spec.jsonl")
PROMPT_OUTPUT_PATH = Path("data/output/prompt_sd35.txt")
CAPTION_OUTPUT_PATH = Path("data/output/caption_sd35.txt")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e

    return records


def get_image_number(record: dict) -> str:
    """
    이미지 번호를 안정적으로 가져온다.

    우선순위:
    1. id
    2. image_filename의 stem
    3. idx
    """
    if record.get("id") is not None:
        return str(record["id"])

    image_filename = record.get("image_filename")
    if image_filename:
        return Path(str(image_filename)).stem

    if record.get("idx") is not None:
        return str(record["idx"])

    return "unknown"


def sort_key(record: dict):
    image_number = get_image_number(record)

    # 000139 같은 숫자형 id는 숫자 기준 정렬
    if image_number.isdigit():
        return (0, int(image_number))

    # 숫자가 아니면 문자열 기준 정렬
    return (1, image_number)


def write_prompt_file(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for record in records:
        image_number = get_image_number(record)
        prompt = str(record.get("image_generation_prompt", "")).strip()

        lines.append(f"===== {image_number} =====")
        lines.append(prompt)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_caption_file(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for record in records:
        image_number = get_image_number(record)

        # 현재 코드에서는 display_caption을 사용함.
        # 혹시 visual_caption이라는 필드를 쓰게 되면 fallback으로 같이 처리.
        caption = str(
            record.get("display_caption")
            or record.get("visual_caption")
            or record.get("caption")
            or ""
        ).strip()

        lines.append(f"===== {image_number} =====")
        lines.append(caption)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    records = load_jsonl(INPUT_PATH)
    records = sorted(records, key=sort_key)

    write_prompt_file(records, PROMPT_OUTPUT_PATH)
    write_caption_file(records, CAPTION_OUTPUT_PATH)

    print(f"[Done] Loaded records: {len(records)}")
    print(f"[Done] Prompt file: {PROMPT_OUTPUT_PATH}")
    print(f"[Done] Caption file: {CAPTION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
