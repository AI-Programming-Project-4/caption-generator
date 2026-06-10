import base64
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, VISION_MODEL


client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Path settings
# =========================

METADATA_PATH = "data/input/metadata.jsonl"

IMAGE_EXCLUDE_DIR = "data/images/images_flowcharts_exclude_image_final_2"
IMAGE_INCLUDE_DIR = "data/images/images_flowcharts_include_image_final_2"

PROMPT_DIR = "prompts"

# SD3.5 Medium 전용 출력 파일
OUTPUT_PATH = "data/output/sd35_generation_dataset_with_spec.jsonl"


# =========================
# Classification settings
# =========================

# OCR label 수 기준 복잡도 분류
# exclude_image: OCR label 8개 이상이면 complex
# include_image: OCR label 7개 이상이면 complex
COMPLEX_OCR_THRESHOLD_EXCLUDE = 8
COMPLEX_OCR_THRESHOLD_INCLUDE = 7

# SD3.5 Medium 전용 프롬프트 파일 매핑
PROMPT_BY_GENERATION_TYPE = {
    "exclude_simple": "caption_prompt_sd35_exclude_simple",
    "exclude_complex": "caption_prompt_sd35_exclude_complex",
    "include_simple": "caption_prompt_sd35_include_simple",
    "include_complex": "caption_prompt_sd35_include_complex",
}

# SD3.5용 positive prompt 길이 제한.
# 실제 tokenizer마다 tokenization이 다를 수 있어, 여기서는 단어/구두점 단위의 보수적 근사값으로 제한한다.
MAX_SD35_PROMPT_TOKENS = 250
MAX_SD35_PROMPT_SENTENCES = 6
MAX_SD35_PROMPT_CHARS = 1200


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
# Metadata helpers
# =========================

def get_ocr_count(record: dict) -> int:
    """
    image_ocr 안의 비어 있지 않은 OCR label 개수를 센다.
    simple/complex 분류는 depth가 아니라 이 OCR 개수를 기준으로 한다.
    """
    image_ocr = record.get("image_ocr", [])

    if not isinstance(image_ocr, list):
        return 0

    return len([x for x in image_ocr if str(x).strip()])


def get_complexity_from_ocr(record: dict) -> str:
    """
    OCR label 수 기준으로 simple/complex를 결정한다.
    include-image figure는 embedded panel이 많아서 threshold를 별도로 둔다.
    """
    flowchart_type = record.get("_flowchart_type", "")
    ocr_count = get_ocr_count(record)

    if flowchart_type == "exclude_image":
        return "complex" if ocr_count >= COMPLEX_OCR_THRESHOLD_EXCLUDE else "simple"

    if flowchart_type == "include_image":
        return "complex" if ocr_count >= COMPLEX_OCR_THRESHOLD_INCLUDE else "simple"

    return "simple"


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
    flowchart_type + OCR count 기반으로 generation_type을 결정한다.
    depth 관련 정보는 사용하지 않는다.
    """
    flowchart_type = record["_flowchart_type"]
    complexity = get_complexity_from_ocr(record)

    if flowchart_type == "exclude_image":
        return f"exclude_{complexity}"

    if flowchart_type == "include_image":
        return f"include_{complexity}"

    return f"exclude_{complexity}"


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
            copied["_ocr_count"] = get_ocr_count(copied)
            copied["_complexity"] = get_complexity_from_ocr(copied)
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


def get_image_sort_key(record: dict) -> str:
    """
    파일명이 000001.png처럼 zero padding 되어 있으면 문자열 정렬만으로 충분하다.
    """
    return extract_filename_from_record(record) or ""


def select_records_for_generation(valid_records: list[dict]) -> list[dict]:
    """
    참고 폴더에 실제로 존재하고, metadata에서 caption이 확인된 모든 이미지를 선택한다.
    유형별 개수 제한은 적용하지 않는다.
    """
    selected = sorted(valid_records, key=get_image_sort_key)

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

def clean_text(value: Any) -> str:
    return str(value).strip()


def compress_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ensure_sentence(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def limit_sentence_count(text: str, max_sentences: int = MAX_SD35_PROMPT_SENTENCES) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= max_sentences:
        return " ".join(parts)
    return " ".join(parts[:max_sentences])


def count_prompt_tokens_approx(text: str) -> int:
    # SD tokenizer와 완전히 같지는 않지만, 단어와 구두점 단위로 보수적 근사값을 계산한다.
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def limit_prompt_tokens_approx(text: str, max_tokens: int = MAX_SD35_PROMPT_TOKENS) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    if len(tokens) <= max_tokens:
        return text

    kept_tokens = tokens[:max_tokens]
    truncated = " ".join(kept_tokens)
    truncated = re.sub(r"\s+([,.;:!?])", r"\1", truncated)
    truncated = compress_whitespace(truncated)

    # 가능하면 마지막 완성 문장까지만 사용한다.
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period >= 80:
        truncated = truncated[: last_period + 1]
    elif truncated and truncated[-1] not in ".!?":
        truncated += "."

    return truncated


def remove_section_labels(prompt: str) -> str:
    """
    SD3.5가 Style:, Text:, Layout: 같은 섹션명을 이미지 안에 그리는 위험을 줄인다.
    """
    section_label_patterns = [
        r"\bStyle\s*:\s*",
        r"\bText\s*:\s*",
        r"\bLabels\s*:\s*",
        r"\bLayout\s*:\s*",
        r"\bLayout and arrows\s*:\s*",
        r"\bPosition and arrows\s*:\s*",
        r"\bArrows\s*:\s*",
        r"\bVisible text\s*:\s*",
    ]

    for pattern in section_label_patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE)

    return prompt


def rewrite_negative_to_positive(prompt: str) -> str:
    """
    LLM이 실수로 출력한 부정형 제약 문장을 positive prompt 표현으로 바꾼다.
    최종 image_generation_prompt에는 가능한 한 allowed-element wording만 남긴다.
    """
    rewrites = {
        r"Do not add a global title or external caption\.?":
            "The canvas starts directly with the figure content and contains only the visible figure area.",
        r"Do not add a new global title or external caption\.?":
            "The canvas starts directly with the figure content and contains only the visible figure area.",
        r"No global title or external caption\.?":
            "The canvas starts directly with the figure content and contains only the visible figure area.",
        r"Do not add any global title, external caption, legend, bullet list, or explanatory paragraph outside the (figure|diagram)\.?":
            "The canvas contains only the visible figure area with clean empty surrounding whitespace.",
        r"Do not add any title, heading, caption, legend, bullet list, explanatory paragraph, or extra text outside the (figure|diagram)\.?":
            "The canvas contains only the visible diagram area with clean empty surrounding whitespace.",
        r"Do not invent extra explanatory text\.?":
            "Visible text consists of original labels and annotations from the figure.",
        r"Do not invent explanatory text outside the figure\.?":
            "Visible text consists of original labels and annotations from the figure.",
        r"Do not invent additional explanatory text\.?":
            "Visible text consists of original labels and annotations from the figure.",
        r"Do not write the (figure|diagram) type itself as visible text\.?":
            "Visible text consists of original labels and annotations from the figure.",
        r"Do not include any explanatory text outside the (figure|diagram)\.?":
            "The surrounding whitespace remains clean and empty.",
        r"The display_caption must not be placed inside the generated image\.?":
            "The generated image contains only the figure content.",
        r"Preserve only labels that belong to visible nodes, arrows, boxes, or annotations\.?":
            "Visible text consists of original node labels, arrow labels, box labels, and annotations.",
        r"Preserve only labels that belong to visible nodes, panels, containers, arrows, or annotations\.?":
            "Visible text consists of original node labels, panel labels, container labels, arrow labels, and annotations.",
        r"Do not replace important visual panels with empty generic boxes\.?":
            "Important embedded visual panels appear as content-bearing visual panels with simplified internal visual content.",
        r"Do not replace important embedded visual panels with empty boxes\.?":
            "Important embedded visual panels appear as content-bearing visual panels with simplified internal visual content.",
        r"If the original (figure|diagram) has no (global )?title, do not create a new title\.?":
            "The canvas starts directly with the figure content.",
        r"without a figure-number prefix":
            "using plain academic caption text",
    }

    for pattern, replacement in rewrites.items():
        prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)

    return prompt


def normalize_sd35_prompt(prompt: str) -> str:
    prompt = clean_text(prompt)

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

    section_headers = [
        "Hard constraints:",
        "Overall layout:",
        "Overall figure type and layout:",
        "Overall architecture layout:",
        "Overall multi-panel layout:",
        "Left or upper region:",
        "Right or lower region:",
        "Center communication or connection region:",
        "Center connection or communication region:",
        "Left panel:",
        "Right panel:",
        "Center region:",
        "Style constraints:",
        "Style:",
        "Text:",
        "Layout:",
        "Layout and arrows:",
        "Position and arrows:",
        "Arrows:",
        "Nodes and positions:",
        "Connections:",
        "Visual regions and embedded panels:",
        "Visual panels and their roles:",
        "Processing blocks and arrows:",
        "Major regions and containers:",
        "Main containers and grouped regions:",
        "Important nodes and module layout:",
        "Major connections and arrow styles:",
        "Visible labels and clean academic style:",
        "Embedded visual panels, visible labels, and clean academic style:",
    ]
    for header in section_headers:
        prompt = prompt.replace(header, "")

    prompt = remove_section_labels(prompt)
    prompt = rewrite_negative_to_positive(prompt)
    prompt = compress_whitespace(prompt)

    if not re.search(r"\bwhite background\b", prompt, flags=re.IGNORECASE):
        prompt = "Clean academic scientific figure on a white background. " + prompt

    if not re.search(r"\bvisible text consists\b", prompt, flags=re.IGNORECASE):
        prompt = (
            prompt.rstrip(". ")
            + ". Visible text consists of original node labels, panel labels, arrow labels, container labels, and annotations."
        )

    if not re.search(r"\bclean empty surrounding whitespace\b", prompt, flags=re.IGNORECASE):
        prompt = (
            prompt.rstrip(". ")
            + ". Clean empty surrounding whitespace around the figure content."
        )

    prompt = compress_whitespace(prompt)
    prompt = limit_sentence_count(prompt, max_sentences=MAX_SD35_PROMPT_SENTENCES)

    if len(prompt) > MAX_SD35_PROMPT_CHARS:
        prompt = prompt[:MAX_SD35_PROMPT_CHARS].rsplit(".", 1)[0] + "."

    prompt = limit_prompt_tokens_approx(prompt, max_tokens=MAX_SD35_PROMPT_TOKENS)
    return prompt


def build_sd35_prompt_from_generation_sections(diagram_spec: dict) -> str:
    """
    diagram_spec.generation_sections를 SD3.5용 자연어 prompt로 변환한다.
    최종 prompt에는 Style:, Text:, Layout: 같은 섹션명을 넣지 않는다.
    """
    sections = diagram_spec.get("generation_sections", {})

    if not isinstance(sections, dict):
        return ""

    style = clean_text(sections.get("style", ""))
    text = clean_text(sections.get("text", ""))
    layout_and_arrows = clean_text(sections.get("layout_and_arrows", ""))

    sentences = []

    if style:
        sentences.append(ensure_sentence(style))

    if text:
        sentences.append(ensure_sentence(text))

    if layout_and_arrows:
        sentences.append(ensure_sentence(layout_and_arrows))

    prompt = " ".join(sentences)
    return normalize_sd35_prompt(prompt)


def clean_image_generation_prompt(prompt: str) -> str:
    """
    fallback용 SD3.5 prompt 정리 함수.
    generation_sections가 없는 경우에만 모델이 반환한 image_generation_prompt를 직접 정리한다.
    """
    return normalize_sd35_prompt(prompt)


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


def validate_and_fill_diagram_spec(diagram_spec: dict, record: dict) -> dict:
    """
    새 spec 구조에서 누락될 수 있는 필드를 최소한의 기본값으로 보강한다.
    OCR 기준 분류를 사용하므로 depth 필드는 spec에 넣지 않는다.
    """
    flowchart_type = record.get("_flowchart_type", "")
    generation_type = record.get("_generation_type", "")
    complexity = record.get("_complexity") or get_complexity_from_ocr(record)

    diagram_spec.setdefault("figure_type", "scientific figure")
    diagram_spec["flowchart_type"] = diagram_spec.get("flowchart_type") or flowchart_type
    diagram_spec["generation_type"] = diagram_spec.get("generation_type") or generation_type
    diagram_spec["complexity"] = diagram_spec.get("complexity") or complexity

    # LLM이 실수로 depth를 반환해도 최종 spec에서는 제거한다.
    diagram_spec.pop("depth", None)

    diagram_spec.setdefault("canvas", {})
    if isinstance(diagram_spec["canvas"], dict):
        diagram_spec["canvas"].setdefault("orientation", diagram_spec.get("orientation", "unknown"))
        diagram_spec["canvas"].setdefault("layout_summary", "")
        diagram_spec["canvas"].setdefault("background", "white")

    diagram_spec.setdefault("style", {})
    if isinstance(diagram_spec["style"], dict):
        diagram_spec["style"].setdefault("overall_style", "clean academic scientific figure")
        diagram_spec["style"].setdefault("color_mode", "black-and-white or minimal color")
        diagram_spec["style"].setdefault("line_style", "thin clean lines")
        diagram_spec["style"].setdefault("font_style", "small readable academic labels")
        diagram_spec["style"].setdefault("geometry_style", "simple aligned geometry")

    diagram_spec.setdefault("text_elements", [])
    diagram_spec.setdefault("layout_elements", [])
    diagram_spec.setdefault("arrows", [])
    diagram_spec.setdefault("embedded_visual_elements", [])
    diagram_spec.setdefault("generation_sections", {})

    if isinstance(diagram_spec["generation_sections"], dict):
        diagram_spec["generation_sections"].setdefault(
            "style",
            "Clean academic scientific figure on a white background with readable labels and aligned geometry.",
        )
        diagram_spec["generation_sections"].setdefault(
            "text",
            "Visible text consists of original node labels, panel labels, arrow labels, container labels, and annotations.",
        )
        diagram_spec["generation_sections"].setdefault(
            "layout_and_arrows",
            "The layout follows the original relative positions with clear arrows between connected elements.",
        )

    return diagram_spec


def parse_json_response(text: str, record: dict | None = None) -> dict:
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

    if record is not None:
        data["diagram_spec"] = validate_and_fill_diagram_spec(
            data["diagram_spec"],
            record=record,
        )

    generation_sections = data["diagram_spec"].get("generation_sections", {})

    if isinstance(generation_sections, dict) and any(
        clean_text(generation_sections.get(key, ""))
        for key in ["style", "text", "layout_and_arrows"]
    ):
        data["image_generation_prompt"] = build_sd35_prompt_from_generation_sections(
            data["diagram_spec"]
        )
    else:
        data["image_generation_prompt"] = clean_image_generation_prompt(
            data["image_generation_prompt"]
        )

    data["display_caption"] = clean_display_caption(
        data["display_caption"]
    )

    return data


def build_negative_prompt(generation_type: str) -> str:
    """
    SD3.5 실행 환경에서 negative_prompt를 별도 입력으로 지원할 때 사용한다.
    이 문자열은 image_generation_prompt 뒤에 이어 붙이지 않는다.
    """
    terms = [
        "global title",
        "external caption",
        "legend",
        "paragraph",
        "bullet list",
        "watermark",
        "signature",
        "decorative background",
        "extra unrelated text",
        "random labels",
        "distorted typography",
    ]

    if generation_type.startswith("exclude"):
        terms.extend([
            "photorealistic scene",
            "decorative illustration",
            "background texture",
        ])

    if generation_type.startswith("include"):
        terms.extend([
            "empty placeholder panel",
            "blank embedded image box",
        ])

    return ", ".join(terms)


# =========================
# Prompt generation
# =========================

def generate_sd35_bundle(record: dict, prompt_text: str) -> dict:
    image_path = record["_resolved_image_path"]
    flowchart_type = record["_flowchart_type"]
    generation_type = record["_generation_type"]
    ocr_count = record.get("_ocr_count", get_ocr_count(record))
    complexity = record.get("_complexity", get_complexity_from_ocr(record))

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
A structured JSON object that describes the figure layout using the required schema below.
Use generation_type={generation_type} and complexity={complexity}.
The caller already selected simple or complex using OCR label count.
Use positive descriptions in all style and generation fields.

2. image_generation_prompt
A compact positive prompt for Stable Diffusion 3.5 Medium.
Keep it under {MAX_SD35_PROMPT_TOKENS} tokens.
Use 3 to 6 concise natural-language sentences.
The prompt should combine generation_sections.style, generation_sections.text, and generation_sections.layout_and_arrows as plain natural language.
Use no section labels such as "Style:", "Text:", "Layout:", or "Arrows:" in image_generation_prompt.
Focus on visible style, readable original labels, relative positions, panels, nodes, and arrows.

3. display_caption
A concise academic caption shown below the generated image as separate metadata.
Use plain caption text.

Paper metadata:
- arXiv ID: {arxiv_id}
- Title: {title}
- Categories: {categories}
- Flowchart type: {flowchart_type}
- Generation type: {generation_type}
- Complexity: {complexity}
- OCR label count: {ocr_count}
- Original paper caption: {original_caption}
- OCR words detected in image: {image_ocr}
- Matched keywords: {matched_keywords}

Type-specific instruction:
{prompt_text}

Output rule:
Return valid JSON only, with exactly these keys:
{{
  "diagram_spec": {{
    "figure_type": "flowchart | architecture | pipeline | multi_panel_figure | scientific_figure",
    "flowchart_type": "{flowchart_type}",
    "generation_type": "{generation_type}",
    "complexity": "{complexity}",
    "canvas": {{
      "orientation": "horizontal | vertical | square | grid | compact | mixed",
      "layout_summary": "...",
      "background": "white"
    }},
    "style": {{
      "overall_style": "...",
      "color_mode": "...",
      "line_style": "...",
      "font_style": "...",
      "geometry_style": "..."
    }},
    "text_elements": [
      {{
        "text": "...",
        "role": "node_label | arrow_label | panel_label | container_label | annotation | resource_label",
        "position": "...",
        "attached_to": "..."
      }}
    ],
    "layout_elements": [
      {{
        "id": "...",
        "type": "node | container | panel | embedded_visual | resource | annotation",
        "label": "...",
        "shape": "rectangle | rounded_rectangle | circle | oval | cylinder | panel | text",
        "position": "...",
        "size": "small | medium | large",
        "contains": []
      }}
    ],
    "arrows": [
      {{
        "source": "...",
        "target": "...",
        "direction": "left_to_right | right_to_left | top_to_bottom | bottom_to_top | bidirectional | diagonal",
        "line_style": "solid | dashed | dotted | curved",
        "label": "",
        "route": "straight | curved | around_container | crossing"
      }}
    ],
    "embedded_visual_elements": [
      {{
        "id": "...",
        "content_type": "plot | waveform | screenshot | illustration | anatomical_figure | result_panel | none",
        "position": "...",
        "role": "...",
        "visual_detail": "..."
      }}
    ],
    "generation_sections": {{
      "style": "One concise sentence describing academic style, background, color mode, lines, fonts, and geometry.",
      "text": "One concise sentence describing the original visible labels to render.",
      "layout_and_arrows": "One to three concise sentences describing relative positions, grouping, panels, node order, and arrows."
    }}
  }},
  "image_generation_prompt": "A plain natural-language SD3.5 prompt under {MAX_SD35_PROMPT_TOKENS} tokens, with no section labels.",
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
                    "Return only valid JSON. Do not output markdown, commentary, or analysis. "
                    "For image_generation_prompt, use compact positive visual wording under 250 tokens. "
                    "Do not include section labels in image_generation_prompt."
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

    return parse_json_response(response.output_text, record=record)


def make_output_record(
    record: dict,
    generated: dict,
    prompt_version: str,
) -> dict:
    image_path = record["_resolved_image_path"]
    flowchart_type = record["_flowchart_type"]
    generation_type = record["_generation_type"]
    ocr_count = record.get("_ocr_count", get_ocr_count(record))
    complexity = record.get("_complexity", get_complexity_from_ocr(record))
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
        "ocr_count": ocr_count,
        "complexity": complexity,
        "original_caption": record.get("caption", ""),
        "diagram_spec": generated["diagram_spec"],
        "image_generation_prompt": generated["image_generation_prompt"],
        "image_generation_prompt_token_count_approx": count_prompt_tokens_approx(
            generated["image_generation_prompt"]
        ),
        "negative_prompt": build_negative_prompt(generation_type),
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
            "prompt_text": load_prompt(prompt_version),
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
        print(f"OCR count: {record.get('_ocr_count')}")
        print(f"Complexity: {record.get('_complexity')}")
        print(f"Generation type: {generation_type}")
        print(f"Prompt version: {prompt_version}")
        print(f"Original caption: {record.get('caption', '')}")

        try:
            generated = generate_sd35_bundle(
                record=record,
                prompt_text=prompt_text,
            )

            output_record = make_output_record(
                record=record,
                generated=generated,
                prompt_version=prompt_version,
            )

            save_jsonl(output_record, OUTPUT_PATH)

            print(f"Display caption: {generated['display_caption']}")
            print(f"Image generation prompt: {generated['image_generation_prompt'][:500]}...")
            print(
                "Approx prompt tokens: "
                f"{output_record['image_generation_prompt_token_count_approx']}"
            )
            print(f"Negative prompt: {output_record['negative_prompt']}")

        except Exception as e:
            print(f"[Error] idx={record.get('idx')}")
            print(f"{type(e).__name__}: {e}")

    print("\n[Done]")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
