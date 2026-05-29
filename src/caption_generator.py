# src/caption_generator.py
from openai import OpenAI
from config import OPENAI_API_KEY, VISION_MODEL
import base64
import json
from pathlib import Path

client = OpenAI(api_key=OPENAI_API_KEY)


def encode_image_to_base64(image_path: str) -> str:
    """
    이미지를 base64 data URL 형태로 변환한다.
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
        raise ValueError("Only png, jpg, jpeg, and webp images are supported.")

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def generate_caption_from_image(
    paper_id: str,
    paper_title: str,
    paper_summary_info: dict,
    image_path: str,
    image_generation_prompt: str,
    figure_number: int = 1,
    language: str = "en"
) -> dict:
    """
    생성된 flowchart 이미지를 분석하고, 논문 내용에 맞는 캡션을 생성한다.
    """

    image_data_url = encode_image_to_base64(image_path)

    schema = {
        "type": "object",
        "properties": {
            "paper_id": {"type": "string"},
            "figure_id": {"type": "string"},
            "image_analysis": {
                "type": "string",
                "description": "Description of visible elements in the generated flowchart image."
            },
            "caption": {
                "type": "string",
                "description": "Final academic caption for the generated flowchart image."
            },
            "mentioned_elements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Elements explicitly mentioned in the caption."
            },
            "missing_or_uncertain_elements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Important paper elements that are not clearly visible in the image."
            },
            "quality_score": {
                "type": "number",
                "description": "Score between 0 and 1 indicating caption-image consistency."
            }
        },
        "required": [
            "paper_id",
            "figure_id",
            "image_analysis",
            "caption",
            "mentioned_elements",
            "missing_or_uncertain_elements",
            "quality_score"
        ],
        "additionalProperties": False
    }

    if language == "ko":
        caption_instruction = """
Write the final caption in Korean.
Start with "Figure {figure_number}."
Use an academic style.
Write one or two concise sentences.
"""
    else:
        caption_instruction = f"""
Write the final caption in English.
Start with "Figure {figure_number}."
Use an academic style.
Write one or two concise sentences.
"""

    user_text = f"""
You are generating a caption for a flowchart image created from a paper or presentation.

Paper ID:
{paper_id}

Paper title:
{paper_title}

Paper summary:
{paper_summary_info.get("paper_summary", "")}

Research goal:
{paper_summary_info.get("research_goal", "")}

Figure purpose:
{paper_summary_info.get("figure_purpose", "")}

Expected key steps:
{paper_summary_info.get("key_steps", [])}

Expected relationships:
{paper_summary_info.get("relationships", [])}

Image generation prompt:
{image_generation_prompt}

Task:
1. Inspect the generated flowchart image.
2. Identify what is actually visible in the image.
3. Write a caption that matches both the image and the paper content.
4. Do not mention elements that are not visible or not supported.
5. If some expected elements are unclear, list them in missing_or_uncertain_elements.

Caption instruction:
{caption_instruction}
"""

    response = client.responses.create(
        model=VISION_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are an academic figure caption generator. "
                    "You must write captions that accurately match the visible image. "
                    "Avoid hallucinating unseen elements."
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
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "flowchart_caption_schema",
                "schema": schema,
                "strict": True
            }
        }
    )

    result = json.loads(response.output_text)
    result["image_path"] = image_path
    result["image_generation_prompt"] = image_generation_prompt
    result["paper_title"] = paper_title
    result["paper_summary_info"] = paper_summary_info

    return result