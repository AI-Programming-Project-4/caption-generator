# prompt_experiment.py
import base64
import json
import re
import math
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from openai import OpenAI

from config import OPENAI_API_KEY, VISION_MODEL


client = OpenAI(api_key=OPENAI_API_KEY)


# 의미 유사도 평가에 사용할 embedding 모델
EMBEDDING_MODEL = "text-embedding-3-small"

# 같은 caption은 embedding API를 다시 호출하지 않도록 메모리 캐시 사용
EMBEDDING_CACHE = {}


# =========================
# Path settings
# =========================

METADATA_PATH = "data/input/metadata.jsonl"

IMAGE_EXCLUDE_DIR = "data/images/images_flowcharts_exclude_image"
IMAGE_INCLUDE_DIR = "data/images/images_flowcharts_include_image"

PROMPT_DIR = "prompts"

OUTPUT_PATH = "data/output/prompt_experiment_results.jsonl"
SUMMARY_PATH = "data/output/prompt_experiment_summary.json"

# 실제 존재하는 이미지 record 중 몇 개만 실험할지 결정
SAMPLES_PER_TYPE = {
    "exclude_image": 50,
    "include_image": 20,
}

SELECTED_PROMPTS = {
    "caption_prompt_v4",
    "caption_prompt_v12",
    "caption_prompt_v14",
    "caption_prompt_v15",
}

# =========================
# File loading
# =========================

def load_jsonl(path: str, max_samples: int | None = None) -> list[dict]:
    """
    metadata.jsonl 파일을 읽는다.
    각 줄은 하나의 JSON 객체여야 한다.
    """
    records = []
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            records.append(record)

            if max_samples is not None and len(records) >= max_samples:
                break

    return records


def load_prompts(prompt_dir: str) -> list[dict]:
    """
    prompts 폴더 안의 .txt 프롬프트 파일들을 읽는다.
    SELECTED_PROMPTS가 설정되어 있으면 해당 프롬프트만 사용한다.
    """
    prompt_path = Path(prompt_dir)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    prompts = []

    for file_path in sorted(prompt_path.glob("*.txt")):
        prompt_version = file_path.stem

        if SELECTED_PROMPTS and prompt_version not in SELECTED_PROMPTS:
            continue

        prompts.append({
            "prompt_version": prompt_version,
            "prompt_text": file_path.read_text(encoding="utf-8").strip()
        })

    if not prompts:
        raise ValueError(
            f"No selected prompt files found in {prompt_dir}. "
            f"Selected prompts: {SELECTED_PROMPTS}"
        )

    return prompts


def save_jsonl(record: dict, output_path: str):
    """
    결과를 JSONL 파일에 append한다.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_output_files():
    """
    같은 실험을 여러 번 실행할 때 결과가 계속 append되는 것을 막고 싶으면 사용한다.
    """
    for path_str in [OUTPUT_PATH, SUMMARY_PATH]:
        path = Path(path_str)
        if path.exists():
            path.unlink()


# =========================
# Image path + type resolver
# =========================

def extract_filename_from_record(record: dict) -> str | None:
    """
    metadata record에서 이미지 파일명만 추출한다.
    예:
    ./arxivcap_filtered_samples\\images\\000106.png
    → 000106.png
    """
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
    """
    metadata.jsonl에서 이미지 파일명만 추출한 뒤,
    실제 이미지가 어느 폴더에 있는지 확인한다.

    기준:
    - data/images/images_flowcharts_exclude_image/{filename}
      → flowchart_type = "exclude_image"

    - data/images/images_flowcharts_include_image/{filename}
      → flowchart_type = "include_image"

    Returns:
        (image_path, flowchart_type)
    """

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
        f"Image file {filename} was not found in either flowchart folder. "
        f"Tried: {exclude_path}, {include_path}"
    )


def filter_existing_image_records(records: list[dict]) -> list[dict]:
    """
    metadata 전체에서 실제 이미지 파일이 존재하는 record만 남긴다.
    이 함수가 중요한 이유:
    - 이미지 번호가 연속적이지 않기 때문
    - metadata 앞에서 10개만 읽으면 실제 이미지 폴더와 안 맞을 수 있음
    """
    valid_records = []
    missing_count = 0
    no_caption_count = 0

    for record in records:
        caption = str(record.get("caption", "")).strip()

        if not caption:
            no_caption_count += 1
            continue

        try:
            image_path, flowchart_type = resolve_image_path_and_type(record)

            copied = dict(record)
            copied["_resolved_image_path"] = image_path
            copied["_flowchart_type"] = flowchart_type
            valid_records.append(copied)

        except FileNotFoundError:
            missing_count += 1

    print("[Filter]")
    print(f"Total metadata records: {len(records)}")
    print(f"Valid records with existing images: {len(valid_records)}")
    print(f"Missing image records: {missing_count}")
    print(f"No caption records: {no_caption_count}")

    type_counts = {}

    for record in valid_records:
        flowchart_type = record["_flowchart_type"]
        type_counts[flowchart_type] = type_counts.get(flowchart_type, 0) + 1

    print(f"Valid records by type: {type_counts}")

    return valid_records


def select_records_for_experiment(valid_records: list[dict]) -> list[dict]:
    """
    flowchart_type별로 지정한 개수만큼 record를 선택한다.

    예:
    SAMPLES_PER_TYPE = {
        "exclude_image": 100,
        "include_image": 30,
    }

    그러면 exclude_image에서 최대 100개,
    include_image에서 최대 30개를 선택한다.
    """
    selected = []

    for flowchart_type, limit in SAMPLES_PER_TYPE.items():
        subset = [
            record for record in valid_records
            if record.get("_flowchart_type") == flowchart_type
        ]

        selected_subset = subset[:limit]
        selected.extend(selected_subset)

        print(
            f"[Select] {flowchart_type}: "
            f"requested={limit}, available={len(subset)}, selected={len(selected_subset)}"
        )

    return selected


def encode_image_to_base64(image_path: str) -> str:
    """
    이미지를 OpenAI API에 넣을 수 있는 base64 data URL로 변환한다.
    """
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
# Text similarity metrics
# =========================

def normalize_text(text: str) -> str:
    """
    비교를 위해 텍스트를 정규화한다.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_f1(reference: str, generated: str) -> float:
    """
    reference caption과 generated caption 사이의 token-level F1.
    단어가 얼마나 겹치는지 보는 보조 지표다.
    """
    ref_tokens = normalize_text(reference).split()
    gen_tokens = normalize_text(generated).split()

    if not ref_tokens or not gen_tokens:
        return 0.0

    ref_set = set(ref_tokens)
    gen_set = set(gen_tokens)

    common = ref_set & gen_set

    if not common:
        return 0.0

    precision = len(common) / len(gen_set)
    recall = len(common) / len(ref_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def sequence_similarity(reference: str, generated: str) -> float:
    """
    문자열 형태와 순서가 얼마나 비슷한지 보는 지표다.
    사용자가 중요하게 보는 기준이므로 combined_score에서 비교적 높은 비중을 둔다.
    """
    ref = normalize_text(reference)
    gen = normalize_text(generated)

    if not ref or not gen:
        return 0.0

    return SequenceMatcher(None, ref, gen).ratio()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    두 embedding vector 사이의 cosine similarity를 계산한다.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def get_text_embedding(text: str) -> list[float]:
    """
    caption 텍스트를 embedding vector로 변환한다.
    같은 텍스트는 캐시해서 API 호출을 줄인다.
    """
    text = text.strip()

    if not text:
        return []

    if text in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[text]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    embedding = response.data[0].embedding
    EMBEDDING_CACHE[text] = embedding

    return embedding


def semantic_similarity_score(reference: str, generated: str) -> float:
    """
    reference caption과 generated caption의 의미적 유사도를 계산한다.
    OpenAI embedding을 사용한다.

    주의:
    - 문자열이 다르더라도 의미가 비슷하면 높은 점수가 나올 수 있다.
    - 예: "System architecture"와 "Overall framework design"
    """
    if not reference.strip() or not generated.strip():
        return 0.0

    ref_embedding = get_text_embedding(reference)
    gen_embedding = get_text_embedding(generated)

    if not ref_embedding or not gen_embedding:
        return 0.0

    score = cosine_similarity(ref_embedding, gen_embedding)

    # 안전하게 0~1 범위로 제한
    return max(0.0, min(1.0, score))


def combined_similarity_score(reference: str, generated: str) -> dict:
    """
    caption 평가 점수를 계산한다.

    핵심 목표:
    - 기존 caption과 generated caption의 의미적 유사도를 가장 중요하게 본다.
    - 사용자가 중요하게 생각하는 sequence_similarity도 높은 비중으로 반영한다.
    - token_f1은 핵심 단어가 겹치는지 확인하는 보조 지표로만 사용한다.

    현재 가중치:
    - semantic_similarity: 55%
    - sequence_similarity: 35%
    - token_f1: 10%

    rouge_l은 단어 순서 겹침 기반이라 의미 평가에서는 제외한다.
    """
    semantic = semantic_similarity_score(reference, generated)
    seq = sequence_similarity(reference, generated)
    f1 = token_f1(reference, generated)

    combined = (
        0.55 * semantic
        + 0.35 * seq
        + 0.10 * f1
    )

    return {
        "semantic_similarity": round(semantic, 4),
        "sequence_similarity": round(seq, 4),
        "token_f1": round(f1, 4),
        "combined_score": round(combined, 4)
    }


# =========================
# Caption generation
# =========================

def generate_caption_with_prompt(
    record: dict,
    image_path: str,
    flowchart_type: str,
    prompt_text: str
) -> str:
    """
    하나의 이미지에 대해 특정 prompt를 사용해 caption을 생성한다.
    """
    image_data_url = encode_image_to_base64(image_path)

    title = record.get("title", "")
    arxiv_id = record.get("arxiv_id", "")
    categories = record.get("categories", "")
    image_ocr = record.get("image_ocr", [])
    matched_keywords = record.get("matched_keywords", [])

    if flowchart_type == "exclude_image":
        type_description = (
            "This is a flowchart mainly composed of boxes, arrows, labels, "
            "and geometric diagram elements. It does not primarily rely on embedded photos."
        )
    elif flowchart_type == "include_image":
        type_description = (
            "This is a flowchart or process diagram that includes embedded images, "
            "such as photos, visual examples, intermediate results, or output images."
        )
    else:
        type_description = "The flowchart type is unknown."

    user_text = f"""
Paper metadata:
- arXiv ID: {arxiv_id}
- Title: {title}
- Categories: {categories}
- OCR words detected in image: {image_ocr}
- Matched keywords: {matched_keywords}
- Flowchart type: {flowchart_type}
- Flowchart type description: {type_description}

Instruction prompt:
{prompt_text}

Task:
Generate a caption for the given image.

Output rule:
Output only the caption text.
"""

    response = client.responses.create(
        model=VISION_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You generate academic figure captions from images. "
                    "You must follow the provided instruction prompt. "
                    "Do not output analysis, only the final caption."
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

    return response.output_text.strip()


# =========================
# Summary
# =========================

def average_scores(subset: list[dict]) -> dict:
    """
    result subset의 평균 점수를 계산한다.
    """
    return {
        "num_samples": len(subset),
        "avg_semantic_similarity": round(
            mean(r["scores"]["semantic_similarity"] for r in subset), 4
        ),
        "avg_sequence_similarity": round(
            mean(r["scores"]["sequence_similarity"] for r in subset), 4
        ),
        "avg_token_f1": round(
            mean(r["scores"]["token_f1"] for r in subset), 4
        ),
        "avg_combined_score": round(
            mean(r["scores"]["combined_score"] for r in subset), 4
        ),
    }

def save_summary(results: list[dict], summary_path: str):
    """
    전체 prompt 성능과 flowchart 유형별 성능을 함께 저장한다.
    """

    prompt_versions = sorted(set(r["prompt_version"] for r in results))
    flowchart_types = sorted(set(r.get("flowchart_type", "unknown") for r in results))

    summary = {
        "overall_by_prompt": {},
        "by_prompt_and_flowchart_type": {},
        "best_prompt_overall": None,
        "best_prompt_by_flowchart_type": {}
    }

    # 1. prompt별 전체 평균
    for version in prompt_versions:
        subset = [r for r in results if r["prompt_version"] == version]
        summary["overall_by_prompt"][version] = average_scores(subset)

    # 2. prompt + flowchart_type별 평균
    for version in prompt_versions:
        summary["by_prompt_and_flowchart_type"][version] = {}

        for flowchart_type in flowchart_types:
            subset = [
                r for r in results
                if r["prompt_version"] == version
                and r.get("flowchart_type", "unknown") == flowchart_type
            ]

            if not subset:
                continue

            summary["by_prompt_and_flowchart_type"][version][flowchart_type] = average_scores(subset)

    # 3. 전체 기준 best prompt
    if summary["overall_by_prompt"]:
        summary["best_prompt_overall"] = max(
            summary["overall_by_prompt"].items(),
            key=lambda item: item[1]["avg_combined_score"]
        )[0]

    # 4. flowchart_type별 best prompt
    for flowchart_type in flowchart_types:
        candidates = []

        for version in prompt_versions:
            type_scores = summary["by_prompt_and_flowchart_type"].get(version, {}).get(flowchart_type)

            if type_scores:
                candidates.append((version, type_scores["avg_combined_score"]))

        if candidates:
            best_version, best_score = max(candidates, key=lambda item: item[1])
            summary["best_prompt_by_flowchart_type"][flowchart_type] = {
                "prompt_version": best_version,
                "avg_combined_score": best_score
            }

    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Prompt Summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def clean_generated_caption(caption: str) -> str:
    caption = caption.strip()

    prefixes = [
        "Figure:",
        "Figure 1:",
        "Figure 1.",
        "Fig.:",
        "Fig. 1:",
        "Caption:"
    ]

    for prefix in prefixes:
        if caption.lower().startswith(prefix.lower()):
            caption = caption[len(prefix):].strip()

    return caption

# =========================
# Main experiment
# =========================

def main():
    # 같은 파일에 계속 누적되는 게 싫으면 주석 해제
    clear_output_files()

    print("[1] Loading metadata...")
    all_records = load_jsonl(METADATA_PATH, max_samples=None)
    print(f"Loaded metadata records: {len(all_records)}")

    print("[2] Filtering records with existing images...")
    valid_records = filter_existing_image_records(all_records)

    print("[3] Selecting records for experiment...")
    records = select_records_for_experiment(valid_records)
    print(f"Selected records: {len(records)}")

    print("[4] Loading prompts...")
    prompts = load_prompts(PROMPT_DIR)
    print(f"Loaded prompts: {[p['prompt_version'] for p in prompts]}")

    all_results = []

    for i, record in enumerate(records):
        ground_truth_caption = record.get("caption", "").strip()

        image_path = record["_resolved_image_path"]
        flowchart_type = record["_flowchart_type"]

        print("=" * 80)
        print(
            f"[Sample {i + 1}/{len(records)}] "
            f"idx={record.get('idx')}, arxiv_id={record.get('arxiv_id')}"
        )
        print(f"Ground truth caption: {ground_truth_caption}")
        print(f"Image path: {image_path}")
        print(f"Flowchart type: {flowchart_type}")

        for prompt in prompts:
            prompt_version = prompt["prompt_version"]
            prompt_text = prompt["prompt_text"]

            print(f"\n[Prompt] {prompt_version}")

            try:
                generated_caption = generate_caption_with_prompt(
                    record=record,
                    image_path=image_path,
                    flowchart_type=flowchart_type,
                    prompt_text=prompt_text
                )

                generated_caption = clean_generated_caption(generated_caption)

                scores = combined_similarity_score(
                    reference=ground_truth_caption,
                    generated=generated_caption
                )

                result = {
                    "idx": record.get("idx"),
                    "arxiv_id": record.get("arxiv_id"),
                    "title": record.get("title"),
                    "categories": record.get("categories"),

                    "image_filename": extract_filename_from_record(record),
                    "image_path": image_path,
                    "flowchart_type": flowchart_type,

                    "ground_truth_caption": ground_truth_caption,
                    "generated_caption": generated_caption,

                    "prompt_version": prompt_version,
                    "prompt_text": prompt_text,

                    "scores": scores,

                    "image_ocr": record.get("image_ocr", []),
                    "matched_keywords": record.get("matched_keywords", [])
                }

                save_jsonl(result, OUTPUT_PATH)
                all_results.append(result)

                print(f"Generated: {generated_caption}")
                print(f"Scores: {scores}")

            except Exception as e:
                print(f"[Error] prompt={prompt_version}, idx={record.get('idx')}")
                print(f"{type(e).__name__}: {e}")

    if all_results:
        save_summary(all_results, SUMMARY_PATH)
    else:
        print("[Warning] No results were generated.")

    print("\n[Done]")
    print(f"Result file: {OUTPUT_PATH}")
    print(f"Summary file: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()