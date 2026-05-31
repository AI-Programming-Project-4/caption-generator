## Qwen-Image 학습용 프롬프트 데이터셋 생성

이 프로젝트에서는 arXiv 기술 논문에서 수집한 flowchart 이미지-캡션 pair를 기반으로, Qwen-Image LoRA 학습 및 프롬프트 개선에 사용할 수 있는 데이터셋을 생성한다.

기존 논문 caption은 `System architecture`, `Overview of the proposed method`, `Processing pipeline`처럼 짧고 추상적인 경우가 많다. 이러한 caption은 사람이 논문 본문과 함께 읽기에는 충분할 수 있지만, text-to-image 모델이 이미지를 생성하기 위한 입력으로 사용하기에는 시각적 정보가 부족하다.

따라서 본 프로젝트에서는 원본 caption을 그대로 사용하지 않고, 이미지와 caption, OCR, metadata를 함께 분석하여 다음 세 가지 정보를 생성한다.

```json
{
  "diagram_spec": {},
  "image_generation_prompt": "...",
  "display_caption": "..."
}
```

각 필드의 역할은 다음과 같다.

| 필드 | 설명 |
|---|---|
| `diagram_spec` | figure의 구조를 정리한 중간 설계도이다. 노드, 컨테이너, 시각 패널, 화살표, 레이아웃 정보를 저장한다. |
| `image_generation_prompt` | Qwen-Image 또는 LoRA 학습에 사용할 실제 이미지 생성 프롬프트이다. |
| `display_caption` | 생성된 이미지 아래에 별도로 보여줄 짧은 설명 caption이다. 이미지 내부에 들어가는 텍스트가 아니다. |

### 왜 `diagram_spec`을 사용하는가?

`diagram_spec`은 Qwen-Image에 직접 입력하는 텍스트가 아니라, 좋은 이미지 생성 프롬프트를 만들기 위한 중간 구조 데이터이다.

원본 caption만으로는 노드의 개수, 배치, 모양, 화살표 방향, 컨테이너 구조, embedded image panel 여부 등을 알기 어렵다. 따라서 먼저 `diagram_spec`을 생성하여 figure의 구조를 명시적으로 정리하고, 이를 바탕으로 `image_generation_prompt`를 만든다.

이 구조는 이후 다음과 같은 확장에도 도움이 된다.

- 원본 이미지 없이 논문 내용이나 사용자 요구만으로 figure를 생성할 때, 먼저 `diagram_spec`을 생성한 뒤 Qwen용 prompt로 변환할 수 있다.
- 사용자가 “오른쪽에 Output module을 추가해줘”, “화살표 방향을 바꿔줘”처럼 피드백을 줄 경우, 긴 prompt 문자열을 직접 수정하는 대신 `diagram_spec`을 수정한 뒤 prompt를 다시 생성할 수 있다.
- 추후 prompt generator를 학습할 때 `논문 문맥 / caption / metadata → diagram_spec → image_generation_prompt` 형태의 학습 데이터로 활용할 수 있다.

### 프롬프트 타입 분리

figure의 유형과 복잡도에 따라 필요한 프롬프트 전략이 다르기 때문에, 현재 프롬프트는 네 가지 타입으로 분리되어 있다.

| 타입 | 설명 |
|---|---|
| `exclude_simple` | 사진이나 embedded image 없이, 간단한 박스와 화살표로 구성된 flowchart 또는 architecture diagram |
| `exclude_complex` | container box, database cylinder, dashed arrow, feedback loop 등 복잡한 구조를 가진 system/software architecture diagram |
| `include_simple` | input/output image, plot, waveform, screenshot 등 일부 embedded visual panel이 포함된 비교적 단순한 figure |
| `include_complex` | Agent 1 / Agent 2 같은 multi-panel 구조, anatomical illustration, waveform, neural map, HMM chain 등 복합 시각 요소가 포함된 figure |

특히 `include_complex`에서는 단순히 “제목을 넣지 말라”고 지시하면 `Agent 1`, `Agent 2`, `Perceptual measures` 같은 실제 panel label이나 annotation까지 금지되는 문제가 생길 수 있다. 따라서 현재 프롬프트에서는 **global title은 금지하되, 원본 figure에 실제로 존재하는 panel label, container label, annotation은 허용**하도록 구분한다.

### 현재 사용하는 코드

현재 Qwen-Image 학습용 프롬프트 데이터셋을 생성하는 주요 스크립트는 다음 파일이다.

```text
build_qwen_generation_dataset_with_spec.py
```

이 스크립트는 다음 과정을 수행한다.

1. `metadata.jsonl` 로드
2. 실제 이미지 파일 존재 여부 확인
3. 이미지가 `exclude_image`인지 `include_image`인지 분류
4. OCR label 수를 기준으로 simple / complex 복잡도 분류
5. 유형에 맞는 prompt 파일 선택
6. GPT Vision API를 이용해 `diagram_spec`, `image_generation_prompt`, `display_caption` 생성
7. 결과를 JSONL 파일로 저장

### 현재 사용하는 프롬프트 파일

프롬프트 파일은 `prompts/` 폴더에 위치한다.

```text
prompts/
├── caption_prompt_qwen_exclude_simple.txt
├── caption_prompt_qwen_exclude_complex.txt
├── caption_prompt_qwen_include_simple.txt
└── caption_prompt_qwen_include_complex.txt
```

각 프롬프트는 figure 유형에 따라 다른 방식으로 구조를 분석하고, Qwen-Image에 넣을 수 있는 상세한 `image_generation_prompt`를 생성하도록 설계되어 있다.

### 입력 데이터 경로

스크립트 상단에서 사용하는 기본 경로는 다음과 같다.

```python
METADATA_PATH = "data/input/metadata.jsonl"

IMAGE_EXCLUDE_DIR = "data/images/images_flowcharts_exclude_image"
IMAGE_INCLUDE_DIR = "data/images/images_flowcharts_include_image"

PROMPT_DIR = "prompts"

OUTPUT_PATH = "data/output/qwen_image_generation_dataset.jsonl"
```

따라서 프로젝트 폴더는 기본적으로 다음 구조를 따른다.

```text
project/
├── main.py
├── build_qwen_generation_dataset_with_spec.py
├── prompts/
│   ├── caption_prompt_qwen_exclude_simple.txt
│   ├── caption_prompt_qwen_exclude_complex.txt
│   ├── caption_prompt_qwen_include_simple.txt
│   └── caption_prompt_qwen_include_complex.txt
├── data/
│   ├── input/
│   │   └── metadata.jsonl
│   ├── images/
│   │   ├── images_flowcharts_exclude_image/
│   │   └── images_flowcharts_include_image/
│   └── output/
└── README.md
```

### 실행 전 설정

API key와 사용할 vision model은 `config.py`에서 관리한다.

예시는 다음과 같다.

```python
OPENAI_API_KEY = "your_api_key"
VISION_MODEL = "gpt-4.1-mini"
```

실제 API key는 GitHub에 업로드하지 않도록 주의해야 한다.  
가능하면 `.env` 파일을 사용하고, `.gitignore`에 `.env`를 추가하는 것을 권장한다.

### 샘플 개수 설정

`build_qwen_generation_dataset_with_spec.py` 안의 `SAMPLES_PER_TYPE` 값을 수정하여 유형별 생성 개수를 조절할 수 있다.

처음 테스트할 때는 적은 개수로 실행하는 것을 권장한다.

```python
SAMPLES_PER_TYPE = {
    "exclude_simple": 3,
    "exclude_complex": 3,
    "include_simple": 2,
    "include_complex": 2,
}
```

품질 확인 후 더 많은 샘플을 생성할 수 있다.

```python
SAMPLES_PER_TYPE = {
    "exclude_simple": 20,
    "exclude_complex": 20,
    "include_simple": 5,
    "include_complex": 5,
}
```

### 실행 방법

프로젝트 루트에서 다음 명령어를 실행한다.

```bash
python build_qwen_generation_dataset_with_spec.py
```

실행이 완료되면 다음 위치에 결과 파일이 생성된다.

```text
data/output/qwen_image_generation_dataset.jsonl
```

### 출력 JSONL 예시

각 줄은 하나의 figure에 대한 전처리 결과를 나타낸다.

```json
{
  "id": "000139",
  "idx": 139,
  "arxiv_id": "example_id",
  "title": "paper title",
  "categories": "cs.AI",
  "image_filename": "000139.png",
  "image_path": "data/images/images_flowcharts_include_image/000139.png",
  "flowchart_type": "include_image",
  "generation_type": "include_complex",
  "original_caption": "Original paper caption",
  "diagram_spec": {
    "figure_type": "...",
    "orientation": "...",
    "major_regions": [],
    "containers": [],
    "nodes": [],
    "embedded_visual_elements": [],
    "connections": [],
    "style_constraints": []
  },
  "image_generation_prompt": "Create only the diagram or figure content...",
  "display_caption": "A concise academic caption...",
  "image_ocr": [],
  "matched_keywords": [],
  "prompt_version": "caption_prompt_qwen_include_complex"
}
```

### Qwen-Image에 실제로 넣는 값

Qwen-Image에 직접 입력하는 텍스트는 `image_generation_prompt`이다.

```json
{
  "text": "image_generation_prompt",
  "image": "original image path"
}
```

`diagram_spec`은 Qwen에 직접 넣는 용도가 아니라 구조 보존, 디버깅, 사용자 피드백 반영, 추후 prompt generator 학습을 위한 중간 데이터이다.

`display_caption`은 생성된 이미지 아래에 보여줄 설명문이며, Qwen-Image 입력 prompt에 섞지 않는 것이 좋다.

### main.py 통합 여부

현재 `build_qwen_generation_dataset_with_spec.py`는 실험용 전처리 스크립트이다.  
아직 prompt 구조와 simple / complex 분류 기준이 개선 중이고, include_complex 유형에 대한 품질 검증도 더 필요하다.

따라서 현재 단계에서는 이 코드를 `main.py`에 합치지 않고 별도 스크립트로 유지하는 것을 권장한다. 추후 데이터 품질 검증, 실패 유형 기록, LoRA 학습 포맷 변환 코드가 안정화되면 `main.py` 또는 별도 CLI entrypoint로 통합할 수 있다.

### 현재 단계

현재 작업은 Qwen-Image 모델을 직접 학습시키는 단계가 아니다.  
현재는 학습과 프롬프트 개선에 사용할 데이터셋을 생성하는 전처리 단계이다.

완료된 내용은 다음과 같다.

- 원본 caption을 그대로 쓰지 않고 이미지 생성용 prompt로 확장하는 구조 설계
- `diagram_spec`, `image_generation_prompt`, `display_caption` 분리
- `exclude_simple`, `exclude_complex`, `include_simple`, `include_complex` 4가지 prompt 타입 분리
- API 기반 자동 전처리 코드 작성
- global title 금지와 panel label / annotation 허용 규칙 정리

추후 보완할 내용은 다음과 같다.

- 실패 유형 기록
- 논문 문맥만으로 `diagram_spec`을 생성하는 단계 추가
