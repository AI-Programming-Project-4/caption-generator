# test_generate_caption.py
from pathlib import Path
import json

from src.paper_processor import extract_paper_summary
from src.caption_generator import generate_caption_from_image
from src.validator import validate_caption_result
from src.dataset_writer import save_jsonl, save_jsonl_if_valid


def load_text_file(text_path: str) -> str:
    path = Path(text_path)

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    return path.read_text(encoding="utf-8").strip()


def main():
    paper_id = "sample_001"
    paper_title = "Flow Chart Image Generator"

    paper_text_path = "data/input/sample_paper.txt"
    image_path = "data/images/sample_flowchart.png"

    image_generation_prompt = """
Create a clean academic flowchart showing the pipeline of the Flow Chart Image Generator project.
The flowchart should include arXiv paper collection, CC BY 4.0 license filtering,
flowchart image-caption pair extraction, caption preprocessing, foundation model search,
LoRA SFT and RLHF training, and final model training.
"""

    print("[0] Loading paper text...")
    paper_text = load_text_file(paper_text_path)

    print("[1] Extracting paper summary...")
    paper_summary_info = extract_paper_summary(
        paper_title=paper_title,
        paper_text=paper_text
    )

    print("[Paper summary]")
    print(json.dumps(paper_summary_info, ensure_ascii=False, indent=2))

    print("\n[2] Generating caption from image...")
    caption_result = generate_caption_from_image(
        paper_id=paper_id,
        paper_title=paper_title,
        paper_summary_info=paper_summary_info,
        image_path=image_path,
        image_generation_prompt=image_generation_prompt,
        figure_number=1,
        language="en"
    )

    print("\n[Caption result]")
    print(json.dumps(caption_result, ensure_ascii=False, indent=2))

    print("\n[3] Validating caption...")
    is_valid, issues = validate_caption_result(caption_result)

    print("is_valid:", is_valid)

    if issues:
        print("issues:")
        for issue in issues:
            print("-", issue)

    print("\n[Final caption]")
    print(caption_result["caption"])

    save_jsonl_if_valid(
        caption_result,
        "data/output/caption_dataset.jsonl"
    )
    print("\n[4] Saved to data/output/caption_dataset.jsonl")


if __name__ == "__main__":
    main()