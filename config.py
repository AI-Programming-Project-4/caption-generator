# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 비용이 부담되면 mini 계열 모델을 먼저 사용해도 됨
TEXT_MODEL = "gpt-4.1-mini"
VISION_MODEL = "gpt-4.1-mini"