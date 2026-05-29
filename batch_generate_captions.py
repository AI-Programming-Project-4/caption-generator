# batch_generate_captions.py
from pathlib import Path
import json
import traceback

from src.paper_processor import extract_paper_summary
from src.caption_generator import generate_caption_from_image
from src.validator import validate_caption_result
from src.dataset_writer import save_jsonl_if_valid, save_jsonl


SAMPLES_PATH = "data/input/samples.json"
OUTPUT_PATH = "data/output/caption_dataset.jsonl"
FAILED_LOG_PATH = "data/output/failed_samples.jsonl"


def load_text_file(text_path: str) -> str:
    """
    paper_text_path에 있는 txt/md 파일을 읽어온다.
    """
    path = Path(text_path)

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    if path.suffix.lower() not in [".txt", ".md"]:
        raise ValueError(f"Only .txt and .md files are supported: {text_path}")

    return path.read_text(encoding="utf-8").strip()


def load_samples(samples_path: str) -> list[dict]:
    """
    samples.json 파일을 읽어온다.
    """
    path = Path(samples_path)

    if not path.exists():
        raise FileNotFoundError(f"samples.json not found: {samples_path}")

    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("samples.json must be a list of sample objects.")

    return samples


def validate_sample_schema(sample: dict, index: int):
    """
    samples.json의 각 sample에 필요한 필드가 있는지 확인한다.
    """
    required_fields = [
        "paper_id",
        "paper_title",
        "paper_text_path",
        "image_path",
        "image_generation_prompt",
    ]

    missing = [field for field in required_fields if field not in sample]

    if missing:
        raise ValueError(
            f"Sample index {index} is missing required fields: {missing}"
        )

    image_path = Path(sample["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {sample['image_path']}")

    text_path = Path(sample["paper_text_path"])
    if not text_path.exists():
        raise FileNotFoundError(f"Paper text file not found: {sample['paper_text_path']}")


def make_failed_record(sample: dict, error: Exception) -> dict:
    """
    실패한 샘플 정보를 failed_samples.jsonl에 저장하기 위한 형태로 만든다.
    """
    return {
        "paper_id": sample.get("paper_id", "unknown"),
        "paper_title": sample.get("paper_title", "unknown"),
        "paper_text_path": sample.get("paper_text_path"),
        "image_path": sample.get("image_path"),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def process_one_sample(sample: dict, index: int, total: int) -> dict:
    """
    sample 하나에 대해:
    paper text 로딩 → summary 추출 → 이미지 캡션 생성 → 검증
    """
    paper_id = sample["paper_id"]
    paper_title = sample["paper_title"]
    paper_text_path = sample["paper_text_path"]
    image_path = sample["image_path"]
    image_generation_prompt = sample["image_generation_prompt"]

    figure_number = sample.get("figure_number", index + 1)
    language = sample.get("language", "en")

    print("=" * 80)
    print(f"[{index + 1}/{total}] Processing: {paper_id}")
    print(f"Title: {paper_title}")

    print("[0] Validating sample schema...")
    validate_sample_schema(sample, index)

    print("[1] Loading paper text...")
    paper_text = load_text_file(paper_text_path)
    print(f"Text length: {len(paper_text)} characters")

    print("[2] Extracting paper summary...")
    paper_summary_info = extract_paper_summary(
        paper_title=paper_title,
        paper_text=paper_text
    )

    print("[3] Generating caption from image...")
    caption_result = generate_caption_from_image(
        paper_id=paper_id,
        paper_title=paper_title,
        paper_summary_info=paper_summary_info,
        image_path=image_path,
        image_generation_prompt=image_generation_prompt,
        figure_number=figure_number,
        language=language
    )

    print("[4] Validating caption...")
    is_valid, issues = validate_caption_result(caption_result)

    caption_result["is_valid"] = is_valid
    caption_result["validation_issues"] = issues
    caption_result["paper_text_path"] = paper_text_path

    print(f"is_valid: {is_valid}")
    print(f"quality_score: {caption_result.get('quality_score')}")

    if issues:
        print("Validation issues:")
        for issue in issues:
            print("-", issue)

    print("[Caption]")
    print(caption_result.get("caption", ""))

    return caption_result


def main():
    print("[Batch] Loading samples...")
    samples = load_samples(SAMPLES_PATH)
    total = len(samples)

    print(f"[Batch] Total samples: {total}")
    print(f"[Batch] Output path: {OUTPUT_PATH}")
    print(f"[Batch] Failed log path: {FAILED_LOG_PATH}")

    success_count = 0
    saved_count = 0
    failed_count = 0
    skipped_count = 0

    for index, sample in enumerate(samples):
        try:
            caption_result = process_one_sample(
                sample=sample,
                index=index,
                total=total
            )

            success_count += 1

            before_saved_count = saved_count

            if caption_result.get("is_valid", False) and caption_result.get("quality_score", 0) >= 0.8:
                save_jsonl_if_valid(caption_result, OUTPUT_PATH)
                saved_count += 1
            else:
                skipped_count += 1
                print("[Skipped] Result was not valid enough to save.")

        except Exception as e:
            failed_count += 1
            print("[ERROR] Failed to process sample.")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")

            failed_record = make_failed_record(sample, e)
            save_jsonl(failed_record, FAILED_LOG_PATH)

    print("=" * 80)
    print("[Batch Finished]")
    print(f"Total samples: {total}")
    print(f"Successfully processed: {success_count}")
    print(f"Saved records: {saved_count}")
    print(f"Skipped records: {skipped_count}")
    print(f"Failed records: {failed_count}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Failed log: {FAILED_LOG_PATH}")


if __name__ == "__main__":
    main()