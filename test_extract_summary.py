# test_extract_summary.py
from pathlib import Path
import json

from src.paper_processor import extract_paper_summary


def load_text_file(text_path: str) -> str:
    path = Path(text_path)

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    return path.read_text(encoding="utf-8").strip()


def main():
    paper_title = "Flow Chart Image Generator"
    paper_text_path = "data/input/sample_paper.txt"

    print("[0] Loading paper text...")
    paper_text = load_text_file(paper_text_path)

    print("\n[Loaded text preview]")
    print("-" * 60)
    print(paper_text[:500])
    print("-" * 60)
    print(f"Text length: {len(paper_text)} characters")

    print("\n[1] Extracting paper summary...")
    paper_summary_info = extract_paper_summary(
        paper_title=paper_title,
        paper_text=paper_text
    )

    print("\n[Result]")
    print(json.dumps(paper_summary_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()