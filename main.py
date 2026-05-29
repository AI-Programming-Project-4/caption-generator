# main.py
from pathlib import Path

from src.paper_processor import extract_paper_summary
from src.caption_generator import generate_caption_from_image
from src.validator import validate_caption_result
from src.dataset_writer import save_jsonl


def load_text_file(text_path: str) -> str:
    """
    input 폴더에 저장된 paper text 파일을 읽어온다.
    """
    path = Path(text_path)

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    if path.suffix.lower() not in [".txt", ".md"]:
        raise ValueError("Only .txt and .md files are supported for paper_text.")

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

    print("Loaded paper text:")
    print("-" * 60)
    print(paper_text[:500])
    print("-" * 60)
    print(f"Text length: {len(paper_text)} characters")

    print("[1] Extracting paper summary...")
    paper_summary_info = extract_paper_summary(
        paper_title=paper_title,
        paper_text=paper_text
    )

    print("[2] Generating caption from image...")
    caption_result = generate_caption_from_image(
        paper_id=paper_id,
        paper_title=paper_title,
        paper_summary_info=paper_summary_info,
        image_path=image_path,
        image_generation_prompt=image_generation_prompt,
        figure_number=1,
        language="en"
    )

    print("[3] Validating caption...")
    is_valid, issues = validate_caption_result(caption_result)

    caption_result["is_valid"] = is_valid
    caption_result["validation_issues"] = issues
    caption_result["paper_text_path"] = paper_text_path

    print("[4] Saving result...")
    save_jsonl(
        caption_result,
        "data/output/caption_dataset.jsonl"
    )

    print("Done.")
    print("Caption:")
    print(caption_result["caption"])

    if not is_valid:
        print("Validation issues:")
        for issue in issues:
            print("-", issue)


if __name__ == "__main__":
    main()