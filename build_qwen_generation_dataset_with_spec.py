import base64
import json
from pathlib import Path

from openai import OpenAI

from config import OPENAI_API_KEY, VISION_MODEL


client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Path settings
# =========================

METADATA_PATH = "data/input/metadata.jsonl"

IMAGE_EXCLUDE_DIR = "data/images/images_flowcharts_exclude_image"
IMAGE_INCLUDE_DIR = "data/images/images_flowcharts_include_image"

PROMPT_DIR = "prompts"

OUTPUT_PATH = "data/output/qwen_image_generation_dataset.jsonl"

# 유형별로 생성할 개수 지정
SAMPLES_PER_TYPE = {
    "exclude_simple": 30,
    "exclude_complex": 30,
    "include_simple": 10,
    "include_complex": 10,
}

# 복잡도 분류 기준
# OCR label 수가 많거나, include_image이면 복잡 구조로 보낼 확률이 높음
COMPLEX_OCR_THRESHOLD_EXCLUDE = 8
COMPLEX_OCR_THRESHOLD_INCLUDE = 6

# 유형별 Qwen 이미지 생성용 프롬프트 생성 instruction 지정
PROMPT_BY_GENERATION_TYPE = {
    "exclude_simple": "caption_prompt_qwen_exclude_simple",
    "exclude_complex": "caption_prompt_qwen_exclude_complex",
    "include_simple": "caption_prompt_qwen_include_simple",
    "include_complex": "caption_prompt_qwen_include_complex",
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
    """
    기존 flowchart_type(exclude/include)에 복잡도(simple/complex)를 더해 generation_type을 만든다.

    기준은 보수적으로 둔다.
    - include_image는 이미지 패널/파형/서브도식이 섞여 복잡할 가능성이 높음
    - OCR label 수가 많으면 복잡 구조로 간주
    """
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


def select_records_for_generation(valid_records: list[dict]) -> list[dict]:
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


def parse_json_response(text: str) -> dict:
    """
    모델이 반환한 JSON 문자열을 dict로 파싱한다.
    혹시 ```json ... ``` 코드블록으로 감싸져 있어도 처리한다.
    """
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

    data["image_generation_prompt"] = clean_image_generation_prompt(
        data["image_generation_prompt"]
    )
    data["display_caption"] = clean_display_caption(
        data["display_caption"]
    )

    if not isinstance(data["diagram_spec"], dict):
        raise ValueError("diagram_spec must be a JSON object.")

    return data


def clean_image_generation_prompt(prompt: str) -> str:
    """
    Qwen에 넣을 이미지 생성 프롬프트 정리.
    이미지 안에 제목/캡션/설명문이 들어가지 않도록 앞뒤로 안전 문구를 보강한다.
    """
    prompt = clean_text(prompt)

    prefixes = [
        "Image generation prompt:",
        "Qwen image generation prompt:",
        "Prompt:",
    ]

    for prefix in prefixes:
        if prompt.lower().startswith(prefix.lower()):
            prompt = prompt[len(prefix):].strip()

    front_rule = (
        "Create only the diagram or figure content on a blank white background. "
        "Do not add any title, heading, caption, legend, bullet list, explanatory paragraph, "
        "or extra text outside the diagram. "
        "Only include text labels that are part of visible nodes, boxes, arrows, panels, or annotations. "
    )

    end_rule = (
        " No title text should appear anywhere in the image. "
        "Do not write the diagram type as visible text. "
        "Do not include any explanatory text outside the diagram. "
        "Prefer structural fidelity over artistic variation."
    )

    return front_rule + prompt + end_rule


def clean_display_caption(caption: str) -> str:
    """
    이미지 아래에 따로 보여줄 caption 정리.
    """
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

def generate_qwen_prompt_bundle(record: dict, prompt_text: str) -> dict:
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

Your task is to create THREE outputs:

1. diagram_spec:
A structured JSON object that describes the figure layout.
It should summarize:
- figure_type
- orientation
- major_regions
- containers
- nodes
- embedded_visual_elements
- connections
- style_constraints

2. image_generation_prompt:
A detailed prompt that will be given directly to Qwen-Image.
It should be based on diagram_spec and should describe the figure in enough visual detail to generate a similar image.
It must not ask Qwen-Image to draw a title, caption, legend, bullet list, or explanatory paragraph inside the image.

3. display_caption:
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
                    "You create structured diagram specifications, Qwen-Image generation prompts, "
                    "and separate display captions for scientific figures. "
                    "Return only valid JSON. Do not output markdown, commentary, or analysis."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_text
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url
                    }
                ]
            }
        ]
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

        # Qwen에 직접 넣을 이미지 생성용 프롬프트
        "image_generation_prompt": generated["image_generation_prompt"],

        # 생성된 이미지 아래에 별도로 보여줄 caption
        "display_caption": generated["display_caption"],

        "image_ocr": record.get("image_ocr", []),
        "matched_keywords": record.get("matched_keywords", []),

        "prompt_version": prompt_version,
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

    print("[4] Loading prompts by generation type...")
    prompt_cache = {}

    for generation_type, prompt_version in PROMPT_BY_GENERATION_TYPE.items():
        prompt_cache[generation_type] = {
            "prompt_version": prompt_version,
            "prompt_text": load_prompt(prompt_version)
        }
        print(f"{generation_type}: {prompt_version}")

    print("[5] Generating diagram specs and Qwen-Image prompts...")

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
            generated = generate_qwen_prompt_bundle(
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
