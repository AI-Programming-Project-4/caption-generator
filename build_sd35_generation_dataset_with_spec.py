import base64
import json
import re
from pathlib import Path

from openai import OpenAI

from config import OPENAI_API_KEY, VISION_MODEL


client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Path settings
# =========================

METADATA_PATH = "data/input/metadata.jsonl"

IMAGE_EXCLUDE_DIR = "data/images/images_flowcharts_exclude_image_final"
IMAGE_INCLUDE_DIR = "data/images/images_flowcharts_include_image_final"

PROMPT_DIR = "prompts"

# SD3.5 Medium 전용 출력 파일
OUTPUT_PATH = "data/output/sd35_generation_dataset_with_spec.jsonl"


# 복잡도 분류 기준
COMPLEX_OCR_THRESHOLD_EXCLUDE = 8
COMPLEX_OCR_THRESHOLD_INCLUDE = 6

# SD3.5 Medium 전용 프롬프트 파일 매핑
PROMPT_BY_GENERATION_TYPE = {
    "exclude_simple": "caption_prompt_sd35_exclude_simple",
    "exclude_complex": "caption_prompt_sd35_exclude_complex",
    "include_simple": "caption_prompt_sd35_include_simple",
    "include_complex": "caption_prompt_sd35_include_complex",
}


# =========================
# File utils
# =========================

def load_jsonl(path: str) -> list[dict]:
    records = []
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    return records


def save_jsonl(record: dict, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_output_file():
    path = Path(OUTPUT_PATH)
    if path.exists():
        path.unlink()


def load_prompt(prompt_version: str) -> str:
    prompt_path = Path(PROMPT_DIR) / f"{prompt_version}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8").strip()


# =========================
# Image resolver
# =========================

def extract_filename_from_record(record: dict) -> str | None:
    raw_image_path = str(record.get("image_path", ""))
    raw_image_file = str(record.get("image_file", ""))

    if raw_image_path:
        normalized = raw_image_path.replace("\\", "/")
        return Path(normalized).name

    if raw_image_file:
        normalized = raw_image_file.replace("\\", "/")
        return Path(normalized).name

    return None


def resolve_image_path_and_type(record: dict) -> tuple[str, str]:
    filename = extract_filename_from_record(record)

    if not filename:
        raise FileNotFoundError(
            f"No image filename found in record idx={record.get('idx')}"
        )

    exclude_path = Path(IMAGE_EXCLUDE_DIR) / filename
    include_path = Path(IMAGE_INCLUDE_DIR) / filename

    if exclude_path.exists():
        return str(exclude_path), "exclude_image"

    if include_path.exists():
        return str(include_path), "include_image"

    raise FileNotFoundError(
        f"Image file {filename} was not found in either folder."
    )


def classify_generation_type(record: dict) -> str:
    flowchart_type = record["_flowchart_type"]
    image_ocr = record.get("image_ocr", [])

    if not isinstance(image_ocr, list):
        image_ocr = []

    ocr_count = len([x for x in image_ocr if str(x).strip()])

    if flowchart_type == "exclude_image":
        if ocr_count >= COMPLEX_OCR_THRESHOLD_EXCLUDE:
            return "exclude_complex"
        return "exclude_simple"

    if flowchart_type == "include_image":
        if ocr_count >= COMPLEX_OCR_THRESHOLD_INCLUDE:
            return "include_complex"
        return "include_simple"

    return "exclude_simple"


def filter_existing_image_records(records: list[dict]) -> list[dict]:
    valid_records = []
    missing_count = 0
    no_caption_count = 0

    for record in records:
        original_caption = str(record.get("caption", "")).strip()

        if not original_caption:
            no_caption_count += 1
            continue

        try:
            image_path, flowchart_type = resolve_image_path_and_type(record)

            copied = dict(record)
            copied["_resolved_image_path"] = image_path
            copied["_flowchart_type"] = flowchart_type
            copied["_generation_type"] = classify_generation_type(copied)
            valid_records.append(copied)

        except FileNotFoundError:
            missing_count += 1

    print("[Filter]")
    print(f"Total metadata records: {len(records)}")
    print(f"Valid records with existing images: {len(valid_records)}")
    print(f"Missing image records: {missing_count}")
    print(f"No caption records: {no_caption_count}")

    type_counts = {}
    generation_counts = {}

    for record in valid_records:
        flowchart_type = record["_flowchart_type"]
        generation_type = record["_generation_type"]

        type_counts[flowchart_type] = type_counts.get(flowchart_type, 0) + 1
        generation_counts[generation_type] = generation_counts.get(generation_type, 0) + 1

    print(f"Valid records by flowchart type: {type_counts}")
    print(f"Valid records by generation type: {generation_counts}")

    return valid_records


#def select_records_for_generation(valid_records: list[dict]) -> list[dict]:
    selected = []

    for generation_type, limit in SAMPLES_PER_TYPE.items():
        subset = [
            record for record in valid_records
            if record.get("_generation_type") == generation_type
        ]

        selected_subset = subset[:limit]
        selected.extend(selected_subset)

        print(
            f"[Select] {generation_type}: "
            f"requested={limit}, available={len(subset)}, selected={len(selected_subset)}"
        )

    return selected
def select_records_for_generation(valid_records: list[dict]) -> list[dict]:
    """
    참고 폴더에 실제로 존재하고, metadata에서 caption이 확인된 모든 이미지를 선택한다.
    유형별 개수 제한은 적용하지 않는다.
    """
    selected = sorted(
        valid_records,
        key=lambda record: str(record.get("image_filename") or record.get("_resolved_image_path") or "")
    )

    generation_counts = {}

    for record in selected:
        generation_type = record.get("_generation_type", "unknown")
        generation_counts[generation_type] = generation_counts.get(generation_type, 0) + 1

    print("[Select] Generate all valid records")
    print(f"Total selected records: {len(selected)}")
    print(f"Selected records by generation type: {generation_counts}")

    return selected

def encode_image_to_base64(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    elif suffix == ".webp":
        mime_type = "image/webp"
    else:
        raise ValueError(f"Unsupported image type: {suffix}")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# =========================
# Response parsing / cleaning
# =========================

def clean_text(value: str) -> str:
    return str(value).strip()


def compress_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def limit_sentence_count(text: str, max_sentences: int = 6) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= max_sentences:
        return " ".join(parts)
    return " ".join(parts[:max_sentences])


def parse_json_response(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {text}") from e

    required_keys = [
        "diagram_spec",
        "image_generation_prompt",
        "display_caption",
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing key in model output: {key}")

    if not isinstance(data["diagram_spec"], dict):
        raise ValueError("diagram_spec must be a JSON object.")

    data["image_generation_prompt"] = clean_image_generation_prompt(
        data["image_generation_prompt"]
    )
    data["display_caption"] = clean_display_caption(
        data["display_caption"]
    )

    return data


def clean_image_generation_prompt(prompt: str) -> str:
    """
    SD3.5 Medium 전용 prompt 정리 함수.

    목표:
    - Qwen 스타일의 긴 구조형 prompt를 더 짧고 압축된 형태로 정리
    - 부정 지시를 줄이고, 핵심 구조/레이아웃/라벨/화살표 정보만 남기기
    - 3~6문장 정도의 자연어 prompt로 유지
    """
    prompt = clean_text(prompt)

    # 앞머리 제거
    prefixes = [
        "Image generation prompt:",
        "Prompt:",
        "Generated prompt:",
        "Stable Diffusion 3.5 Medium prompt:",
        "SD3.5 prompt:",
    ]
    for prefix in prefixes:
        if prompt.lower().startswith(prefix.lower()):
            prompt = prompt[len(prefix):].strip()

    # Qwen 스타일 섹션 헤더 제거
    section_headers = [
        "Hard constraints:",
        "Overall layout:",
        "Left or upper region:",
        "Right or lower region:",
        "Center communication or connection region:",
        "Left panel:",
        "Right panel:",
        "Center region:",
        "Style constraints:",
        "Style:",
        "Nodes and positions:",
        "Connections:",
        "Visual regions and embedded panels:",
        "Processing blocks and arrows:",
        "Major regions and containers:",
        "Main containers and grouped regions:",
        "Important nodes and module layout:",
        "Major connections and arrow styles:",
        "Visual panels and their roles:",
        "Overall figure type and layout:",
    ]

    for header in section_headers:
        prompt = prompt.replace(header, "")

    # 과한 부정 지시 정리
    long_negative_patterns = [
        r"Do not add any global title, external caption, legend, bullet list, or explanatory paragraph outside the (figure|diagram)\.?",
        r"Do not add any title, heading, caption, legend, bullet list, explanatory paragraph, or extra text outside the (figure|diagram)\.?",
        r"Do not write the (figure|diagram) type itself as visible text\.?",
        r"Do not invent extra explanatory text\.?",
        r"Do not include any explanatory text outside the (figure|diagram)\.?",
        r"The display_caption must not be placed inside the generated image\.?",
        r"Preserve only labels that belong to visible nodes, panels, containers, arrows, or annotations\.?",
    ]
    for pattern in long_negative_patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE)

    # 중복 문장 정리
    prompt = compress_whitespace(prompt)

    # 문장 보강
    if not re.search(r"\bwhite background\b", prompt, flags=re.IGNORECASE):
        prompt = "A clean academic figure on a white background. " + prompt

    if not re.search(r"\bNo global title or external caption\b", prompt, flags=re.IGNORECASE):
        prompt = prompt.rstrip(". ") + ". No global title or external caption."

    prompt = compress_whitespace(prompt)
    prompt = limit_sentence_count(prompt, max_sentences=6)

    # 길이 제한
    max_chars = 900
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars].rsplit(".", 1)[0] + "."

    return prompt


def clean_display_caption(caption: str) -> str:
    caption = clean_text(caption)

    prefixes = [
        "Figure:",
        "Figure 1:",
        "Figure 1.",
        "Fig.:",
        "Fig. 1:",
        "Caption:",
        "Display caption:",
    ]

    for prefix in prefixes:
        if caption.lower().startswith(prefix.lower()):
            caption = caption[len(prefix):].strip()

    return caption


# =========================
# Prompt generation
# =========================

def generate_sd35_bundle(record: dict, prompt_text: str) -> dict:
    image_path = record["_resolved_image_path"]
    flowchart_type = record["_flowchart_type"]
    generation_type = record["_generation_type"]

    image_data_url = encode_image_to_base64(image_path)

    original_caption = record.get("caption", "")
    title = record.get("title", "")
    arxiv_id = record.get("arxiv_id", "")
    categories = record.get("categories", "")
    image_ocr = record.get("image_ocr", [])
    matched_keywords = record.get("matched_keywords", [])

    user_text = f"""
You are given an existing scientific image-caption pair.

Your task is to create THREE outputs for Stable Diffusion 3.5 Medium:

1. diagram_spec
A structured JSON object that describes the figure layout.

2. image_generation_prompt
A prompt that will be given directly to Stable Diffusion 3.5 Medium.
This prompt must be shorter and more compact than a Qwen-style prompt.
Use concise structured natural language.
Prefer about 3 to 6 sentences.
Focus on the main layout, important nodes or panels, major arrows, and readable labels.
Avoid long negative instruction lists.
Use at most one short negative sentence such as:
"No global title or external caption."

3. display_caption
A concise academic caption that will be shown below the generated image.
It must not be drawn inside the image.

Paper metadata:
- arXiv ID: {arxiv_id}
- Title: {title}
- Categories: {categories}
- Flowchart type: {flowchart_type}
- Generation type: {generation_type}
- Original paper caption: {original_caption}
- OCR words detected in image: {image_ocr}
- Matched keywords: {matched_keywords}

Type-specific instruction:
{prompt_text}

Output rule:
Return valid JSON only, with exactly these keys:
{{
  "diagram_spec": {{
    "figure_type": "...",
    "orientation": "...",
    "major_regions": [],
    "containers": [],
    "nodes": [],
    "embedded_visual_elements": [],
    "connections": [],
    "style_constraints": []
  }},
  "image_generation_prompt": "...",
  "display_caption": "..."
}}
"""

    response = client.responses.create(
        model=VISION_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You create structured diagram specifications, Stable Diffusion 3.5 Medium generation prompts, "
                    "and separate display captions for scientific figures. "
                    "Return only valid JSON. Do not output markdown, commentary, or analysis."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_text,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                ],
            },
        ],
    )

    return parse_json_response(response.output_text)


def make_output_record(
    record: dict,
    generated: dict,
    prompt_version: str
) -> dict:
    image_path = record["_resolved_image_path"]
    flowchart_type = record["_flowchart_type"]
    generation_type = record["_generation_type"]
    filename = extract_filename_from_record(record)
    image_id = Path(filename).stem if filename else str(record.get("idx"))

    return {
        "id": image_id,
        "idx": record.get("idx"),
        "arxiv_id": record.get("arxiv_id"),
        "title": record.get("title"),
        "categories": record.get("categories"),

        "image_filename": filename,
        "image_path": image_path,
        "flowchart_type": flowchart_type,
        "generation_type": generation_type,

        "original_caption": record.get("caption", ""),

        "diagram_spec": generated["diagram_spec"],

        # SD3.5 Medium에 직접 넣을 프롬프트
        "image_generation_prompt": generated["image_generation_prompt"],

        # 생성된 이미지 아래에 따로 보여줄 캡션
        "display_caption": generated["display_caption"],

        "image_ocr": record.get("image_ocr", []),
        "matched_keywords": record.get("matched_keywords", []),

        "prompt_version": prompt_version,
        "target_model": "sd3.5-medium",
    }


# =========================
# Main
# =========================

def main():
    clear_output_file()

    print("[1] Loading metadata...")
    all_records = load_jsonl(METADATA_PATH)
    print(f"Loaded metadata records: {len(all_records)}")

    print("[2] Filtering records with existing images...")
    valid_records = filter_existing_image_records(all_records)

    print("[3] Selecting records for generation...")
    selected_records = select_records_for_generation(valid_records)
    print(f"Selected records: {len(selected_records)}")

    print("[4] Loading SD3.5 prompt files...")
    prompt_cache = {}

    for generation_type, prompt_version in PROMPT_BY_GENERATION_TYPE.items():
        prompt_cache[generation_type] = {
            "prompt_version": prompt_version,
            "prompt_text": load_prompt(prompt_version)
        }
        print(f"{generation_type}: {prompt_version}")

    print("[5] Generating diagram specs and SD3.5 prompts...")

    for i, record in enumerate(selected_records):
        generation_type = record["_generation_type"]

        if generation_type not in prompt_cache:
            print(f"[Skip] No prompt configured for generation_type={generation_type}")
            continue

        prompt_version = prompt_cache[generation_type]["prompt_version"]
        prompt_text = prompt_cache[generation_type]["prompt_text"]

        print("=" * 80)
        print(
            f"[Sample {i + 1}/{len(selected_records)}] "
            f"idx={record.get('idx')}, arxiv_id={record.get('arxiv_id')}"
        )
        print(f"Flowchart type: {record['_flowchart_type']}")
        print(f"Generation type: {generation_type}")
        print(f"Prompt version: {prompt_version}")
        print(f"Original caption: {record.get('caption', '')}")

        try:
            generated = generate_sd35_bundle(
                record=record,
                prompt_text=prompt_text
            )

            output_record = make_output_record(
                record=record,
                generated=generated,
                prompt_version=prompt_version
            )

            save_jsonl(output_record, OUTPUT_PATH)

            print(f"Display caption: {generated['display_caption']}")
            print(f"Image generation prompt: {generated['image_generation_prompt'][:500]}...")

        except Exception as e:
            print(f"[Error] idx={record.get('idx')}")
            print(f"{type(e).__name__}: {e}")

    print("\n[Done]")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
