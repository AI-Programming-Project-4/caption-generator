# SD3.5 Medium용 Flowchart 프롬프트 데이터셋 생성

이 프로젝트에서는 arXiv 기술 논문에서 수집한 flowchart 이미지-캡션 pair를 기반으로, **Stable Diffusion 3.5 Medium(sd35-medium)**을 이용해 scientific flowchart / architecture diagram 이미지를 생성하기 위한 프롬프트 데이터셋을 만든다.

기존 논문 caption은 `System architecture`, `Overview of the proposed method`, `Processing pipeline`처럼 짧고 추상적인 경우가 많다. 이러한 caption은 사람이 논문 본문과 함께 읽기에는 충분할 수 있지만, text-to-image 모델이 이미지를 생성하기 위한 입력으로 사용하기에는 시각적 정보가 부족하다.

따라서 본 프로젝트에서는 원본 caption을 그대로 사용하지 않고, 이미지와 caption, OCR, metadata를 함께 분석하여 다음 정보를 생성한다.

```json
{
  "diagram_spec": {},
  "image_generation_prompt": "...",
  "negative_prompt": "...",
  "display_caption": "..."
}
```

각 필드의 역할은 다음과 같다.

| 필드 | 설명 |
|---|---|
| `diagram_spec` | figure의 구조를 정리한 중간 설계도이다. 스타일, 들어갈 텍스트, 노드/패널 위치, 화살표 정보를 구조화해서 저장한다. |
| `image_generation_prompt` | sd35-medium에 직접 입력할 positive prompt이다. `diagram_spec.generation_sections`를 바탕으로 자연어 문장으로 재구성된다. |
| `negative_prompt` | sd35-medium 실행 환경에서 negative prompt 입력을 별도로 지원할 때 사용하는 텍스트이다. positive prompt 뒤에 이어 붙이지 않는다. |
| `display_caption` | 생성된 이미지 아래에 별도로 보여줄 짧은 설명 caption이다. 이미지 내부에 들어가는 텍스트가 아니다. |

## 왜 `diagram_spec`을 사용하는가?

`diagram_spec`은 sd35-medium에 직접 입력하는 텍스트가 아니라, 좋은 이미지 생성 프롬프트를 만들기 위한 중간 구조 데이터이다.

원본 caption만으로는 노드의 개수, 배치, 모양, 화살표 방향, 컨테이너 구조, embedded image panel 여부 등을 알기 어렵다. 따라서 먼저 `diagram_spec`을 생성하여 figure의 구조를 명시적으로 정리하고, 이를 바탕으로 `image_generation_prompt`를 만든다.

특히 현재 구조에서는 `generation_sections`를 사용해 프롬프트 정보를 다음 세 가지로 통일한다.

```json
{
  "generation_sections": {
    "style": "...",
    "text": "...",
    "layout_and_arrows": "..."
  }
}
```

- `style`: 흰 배경, academic style, 선 스타일, 폰트, 도형 스타일 등
- `text`: 이미지 안에 들어갈 원본 label, panel label, arrow label, annotation 등
- `layout_and_arrows`: 노드/패널의 상대 위치, 그룹 구조, 흐름 방향, 화살표 정보 등

다만 sd35-medium에 넣는 최종 `image_generation_prompt`에는 `Style:`, `Text:`, `Layout:` 같은 섹션명을 그대로 넣지 않는다. 이러한 단어가 이미지 안에 렌더링될 위험이 있기 때문에, 코드에서 `generation_sections`를 자연어 문장으로 합쳐 최종 prompt를 만든다.

예시는 다음과 같다.

```text
Clean black-and-white academic diagram on a white background with thin lines and readable labels. Visible text consists of the original labels: Input, Encoder, Decoder, Output. The layout is horizontal from left to right, with rectangular nodes connected by solid arrows.
```

## 프롬프트 타입 분리

figure의 유형과 복잡도에 따라 필요한 프롬프트 전략이 다르기 때문에, 현재 프롬프트는 네 가지 타입으로 분리되어 있다.

| 타입 | 설명 |
|---|---|
| `exclude_simple` | 사진이나 embedded image 없이, 간단한 박스와 화살표로 구성된 flowchart 또는 architecture diagram |
| `exclude_complex` | container box, database cylinder, dashed arrow, feedback loop 등 복잡한 구조를 가진 system/software architecture diagram |
| `include_simple` | input/output image, plot, waveform, screenshot 등 일부 embedded visual panel이 포함된 비교적 단순한 figure |
| `include_complex` | Agent 1 / Agent 2 같은 multi-panel 구조, anatomical illustration, waveform, neural map, HMM chain 등 복합 시각 요소가 포함된 figure |

## simple / complex 분류 기준

현재 simple / complex 분류는 **OCR label 수**를 기준으로 한다.

```python
COMPLEX_OCR_THRESHOLD_EXCLUDE = 8
COMPLEX_OCR_THRESHOLD_INCLUDE = 7
```

기본 기준은 다음과 같다.

| flowchart type | simple | complex |
|---|---:|---:|
| `exclude_image` | OCR label 수 < 8 | OCR label 수 >= 8 |
| `include_image` | OCR label 수 < 7 | OCR label 수 >= 7 |

`include_image` 기준을 더 낮추고 싶으면 다음 값만 수정하면 된다.

```python
COMPLEX_OCR_THRESHOLD_INCLUDE = 6
```

## 현재 사용하는 코드

현재 sd35-medium용 프롬프트 데이터셋을 생성하는 주요 스크립트는 다음 파일이다.

```text
build_sd35_generation_dataset_with_spec.py
```

이 스크립트는 다음 과정을 수행한다.

1. `metadata.jsonl` 로드
2. 실제 이미지 파일 존재 여부 확인
3. 이미지가 `exclude_image`인지 `include_image`인지 분류
4. OCR label 수를 기준으로 simple / complex 복잡도 분류
5. 유형에 맞는 sd35 prompt 파일 선택
6. GPT Vision API를 이용해 `diagram_spec`, `image_generation_prompt`, `display_caption` 생성
7. `diagram_spec.generation_sections`를 이용해 sd35용 positive prompt를 250 토큰 이하 자연어 문장으로 재구성
8. 결과를 JSONL 파일로 저장

## 현재 사용하는 프롬프트 파일

프롬프트 파일은 `prompts/` 폴더에 위치한다.

```text
prompts/
├── caption_prompt_sd35_exclude_simple.txt
├── caption_prompt_sd35_exclude_complex.txt
├── caption_prompt_sd35_include_simple.txt
└── caption_prompt_sd35_include_complex.txt
```

각 프롬프트는 figure 유형에 따라 다른 방식으로 구조를 분석하고, sd35-medium에 적합한 짧고 직접적인 `image_generation_prompt`를 생성하도록 설계되어 있다.

현재 prompt 파일의 주요 규칙은 다음과 같다.

- `diagram_spec`은 고정된 구조를 따른다.
- `generation_sections.style`, `generation_sections.text`, `generation_sections.layout_and_arrows`를 반드시 채운다.
- 최종 `image_generation_prompt`는 250 토큰 이하로 유지한다.
- 최종 `image_generation_prompt`에는 `Style:`, `Text:`, `Layout:`, `Arrows:` 같은 섹션명을 넣지 않는다.
- 최종 `image_generation_prompt`는 부정문보다 긍정형/허용목록 표현을 사용한다.
- 실제 figure에 존재하는 node label, panel label, container label, arrow label, annotation은 보존한다.

## 입력 데이터 경로

스크립트 상단에서 사용하는 기본 경로는 다음과 같다.

```python
METADATA_PATH = "data/input/metadata.jsonl"

IMAGE_EXCLUDE_DIR = "data/images/images_flowcharts_exclude_image_final"
IMAGE_INCLUDE_DIR = "data/images/images_flowcharts_include_image_final"

PROMPT_DIR = "prompts"

OUTPUT_PATH = "data/output/sd35_generation_dataset_with_spec.jsonl"
```

따라서 프로젝트 폴더는 기본적으로 다음 구조를 따른다.

```text
project/
├── build_sd35_generation_dataset_with_spec.py
├── config.py
├── prompts/
│   ├── caption_prompt_sd35_exclude_simple.txt
│   ├── caption_prompt_sd35_exclude_complex.txt
│   ├── caption_prompt_sd35_include_simple.txt
│   └── caption_prompt_sd35_include_complex.txt
├── data/
│   ├── input/
│   │   └── metadata.jsonl
│   ├── images/
│   │   ├── images_flowcharts_exclude_image_final/
│   │   └── images_flowcharts_include_image_final/
│   └── output/
└── README.md
```

## 실행 전 설정

API key와 사용할 vision model은 `config.py`에서 관리한다.

예시는 다음과 같다.

```python
OPENAI_API_KEY = "your_api_key"
VISION_MODEL = "gpt-4.1-mini"
```

실제 API key는 GitHub에 업로드하지 않도록 주의해야 한다. 가능하면 `.env` 파일을 사용하고, `.gitignore`에 `.env`를 추가하는 것을 권장한다.

## 생성 개수 설정

현재 스크립트는 참고 폴더에 실제로 존재하고, metadata에서 caption이 확인된 모든 이미지를 대상으로 생성한다. 즉, 이전처럼 `SAMPLES_PER_TYPE`으로 유형별 개수를 제한하지 않는다.

테스트용으로 일부만 실행하고 싶으면 `select_records_for_generation()`에서 슬라이싱을 추가하면 된다.

```python
def select_records_for_generation(valid_records: list[dict]) -> list[dict]:
    selected = sorted(valid_records, key=get_image_sort_key)
    return selected[:20]
```

## 실행 방법

프로젝트 루트에서 다음 명령어를 실행한다.

```bash
python build_sd35_generation_dataset_with_spec.py
```

실행이 완료되면 다음 위치에 결과 파일이 생성된다.

```text
data/output/sd35_generation_dataset_with_spec.jsonl
```

## 출력 JSONL 예시

각 줄은 하나의 figure에 대한 전처리 결과를 나타낸다.

```json
{
  "id": "000139",
  "idx": 139,
  "arxiv_id": "example_id",
  "title": "paper title",
  "categories": "cs.AI",
  "image_filename": "000139.png",
  "image_path": "data/images/images_flowcharts_include_image_final/000139.png",
  "flowchart_type": "include_image",
  "generation_type": "include_complex",
  "ocr_count": 12,
  "complexity": "complex",
  "original_caption": "Original paper caption",
  "diagram_spec": {
    "figure_type": "multi_panel_figure",
    "flowchart_type": "include_image",
    "generation_type": "include_complex",
    "complexity": "complex",
    "canvas": {
      "orientation": "horizontal",
      "layout_summary": "side-by-side multi-panel layout",
      "background": "white"
    },
    "style": {
      "overall_style": "clean academic scientific figure",
      "color_mode": "black-and-white or minimal color",
      "line_style": "thin clean lines",
      "font_style": "small readable academic labels",
      "geometry_style": "simple aligned geometry"
    },
    "text_elements": [],
    "layout_elements": [],
    "arrows": [],
    "embedded_visual_elements": [],
    "generation_sections": {
      "style": "Clean academic paper figure on a white background with readable labels and aligned panels.",
      "text": "Visible text consists of original panel labels, node labels, arrow labels, and annotations.",
      "layout_and_arrows": "The layout follows the original panel positions with clear arrows between connected elements."
    }
  },
  "image_generation_prompt": "Clean academic paper figure on a white background with readable labels and aligned panels. Visible text consists of original panel labels, node labels, arrow labels, and annotations. The layout follows the original panel positions with clear arrows between connected elements.",
  "image_generation_prompt_token_count_approx": 39,
  "negative_prompt": "global title, external caption, legend, paragraph, bullet list, watermark, signature, decorative background, extra unrelated text, random labels, distorted typography, empty placeholder panel, blank embedded image box",
  "display_caption": "A concise academic caption...",
  "image_ocr": [],
  "matched_keywords": [],
  "prompt_version": "caption_prompt_sd35_include_complex",
  "target_model": "sd3.5-medium"
}
```

## sd35-medium에 실제로 넣는 값

sd35-medium에 직접 입력하는 positive prompt는 `image_generation_prompt`이다.

```json
{
  "prompt": "image_generation_prompt",
  "negative_prompt": "negative_prompt"
}
```

`diagram_spec`은 sd35-medium에 직접 넣는 용도가 아니라 구조 보존, 디버깅, 사용자 피드백 반영, 추후 prompt generator 학습을 위한 중간 데이터이다.

`display_caption`은 생성된 이미지 아래에 보여줄 설명문이며, sd35-medium 입력 prompt에 섞지 않는 것이 좋다.

## main.py 통합 여부

현재 `build_sd35_generation_dataset_with_spec.py`는 실험용 전처리 스크립트이다. 아직 prompt 구조, OCR 기준 simple / complex 분류 기준, include_complex 유형에 대한 품질 검증은 더 조정될 수 있다.

따라서 현재 단계에서는 이 코드를 `main.py`에 합치지 않고 별도 스크립트로 유지하는 것을 권장한다. 추후 데이터 품질 검증, 실패 유형 기록, LoRA 학습 포맷 변환 코드가 안정화되면 `main.py` 또는 별도 CLI entrypoint로 통합할 수 있다.

## 현재 단계

현재 작업은 sd35-medium으로 flowchart 이미지를 생성하기 위한 프롬프트 데이터셋을 만드는 전처리 단계이다.

완료된 내용은 다음과 같다.

- 원본 caption을 그대로 쓰지 않고 이미지 생성용 prompt로 확장하는 구조 설계
- `diagram_spec`, `image_generation_prompt`, `negative_prompt`, `display_caption` 분리
- `generation_sections.style`, `generation_sections.text`, `generation_sections.layout_and_arrows` 기반 prompt 통일
- `exclude_simple`, `exclude_complex`, `include_simple`, `include_complex` 4가지 prompt 타입 분리
- OCR label 수 기반 simple / complex 분류
- sd35-medium용 positive prompt를 250 토큰 이하로 제한
- `Style:`, `Text:`, `Layout:` 같은 섹션명을 최종 prompt에서 제거
- global title / external caption 관련 내용은 negative prompt로 분리

추후 보완할 내용은 다음과 같다.

- 실패 유형 기록
- OCR 기준 threshold 조정 실험
- 생성 이미지 품질 평가 스크립트 추가
- 논문 문맥만으로 `diagram_spec`을 생성하는 단계 추가
- sd35-medium 학습 또는 LoRA 학습 포맷 변환 코드 추가
