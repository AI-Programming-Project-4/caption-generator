# src/paper_processor.py
from openai import OpenAI
from config import OPENAI_API_KEY, TEXT_MODEL
import json

client = OpenAI(api_key=OPENAI_API_KEY)


def extract_paper_summary(paper_title: str, paper_text: str) -> dict:
    """
    논문/프레젠테이션 텍스트에서 flowchart 캡션 생성에 필요한 핵심 정보만 추출한다.
    """

    schema = {
        "type": "object",
        "properties": {
            "paper_summary": {
                "type": "string",
                "description": "Short summary of the paper or presentation material."
            },
            "research_goal": {
                "type": "string",
                "description": "Main goal of the project or paper."
            },
            "figure_purpose": {
                "type": "string",
                "description": "What the flowchart figure is intended to explain."
            },
            "key_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Main steps that should appear in the flowchart."
            },
            "relationships": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relationships between steps."
            }
        },
        "required": [
            "paper_summary",
            "research_goal",
            "figure_purpose",
            "key_steps",
            "relationships"
        ],
        "additionalProperties": False
    }

    prompt = f"""
You are a research paper analysis assistant.

Extract only the information needed to write a caption for a flowchart figure.

Paper title:
{paper_title}

Paper or presentation text:
{paper_text}

Return structured information only.
"""

    response = client.responses.create(
        model=TEXT_MODEL,
        input=[
            {
                "role": "system",
                "content": "You extract structured information from academic or project texts."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "paper_summary_schema",
                "schema": schema,
                "strict": True
            }
        }
    )

    return json.loads(response.output_text)