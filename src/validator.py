# src/validator.py

def validate_caption_result(result: dict) -> tuple[bool, list[str]]:
    """
    생성된 캡션 결과가 기본 조건을 만족하는지 검사한다.
    """

    issues = []

    caption = result.get("caption", "")
    quality_score = result.get("quality_score", 0)

    if not caption:
        issues.append("Caption is empty.")

    if len(caption) < 40:
        issues.append("Caption is too short.")

    if len(caption) > 600:
        issues.append("Caption is too long.")

    if not caption.lower().startswith("figure"):
        issues.append("Caption should start with 'Figure'.")

    if quality_score < 0.7:
        issues.append(f"Quality score is low: {quality_score}")

    mentioned_elements = result.get("mentioned_elements", [])
    if len(mentioned_elements) < 2:
        issues.append("Too few mentioned elements.")

    return len(issues) == 0, issues