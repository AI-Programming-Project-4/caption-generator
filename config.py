# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TEXT_MODEL = "gpt-4.1-mini"
VISION_MODEL = "gpt-4.1-mini"
