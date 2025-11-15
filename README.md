#  동아리 내규 안내 AI 챗봇 (RAG + GGUF/LLM)

동아리 신입 부원들의 반복적인 질문에 자동으로 응답하여, 기존 부원들의 업무 효율을 높이기 위해 개발한 RAG(검색 증강 생성) 기반 AI 챗봇입니다.

PDF 형태의 동아리 내규를 기반으로, 사용자의 질문에 대해 24시간 정확한 답변을 제공하는 것을 목표로 합니다.

## 🖼️ 데모 스크린샷


<img width="1091" height="808" alt="image" src="https://github.com/user-attachments/assets/4e77b0e0-143c-4556-83e7-cda1284cbd34" />


---

##  핵심 기능

* **RAG(검색 증강 생성):** '동아리 내규' PDF 문서의 내용을 기반으로만 답변하여, LLM의 환각(Hallucination)을 방지하고 정보의 정확성을 보장합니다.
* **실시간 채팅 UI:** Streamlit을 활용하여 사용자가 실시간으로 질문하고 답변을 받을 수 있는 인터랙티브 웹 인터페이스를 제공합니다.
* **100% 로컬 구동 (No API):** 모든 임베딩 모델과 LLM을 API 호출 없이 서버에 직접 배포하여, API 비용과 네트워크 지연 없이 100% 무료로 운영됩니다.

---

##  사용 기술 및 아키텍처

### 1. Tech Stack

* **Frontend:** `Streamlit`
* **Backend (RAG Pipeline):** `LangChain`
* **LLM (Brain):** `google/gemma-2b-it` (GGUF / Q4_K_M 경량화 버전)
* **LLM Runner:** `llama-cpp-python`
* **Embedding Model:** `jhgan/ko-sbert-nli` (한국어 특화 임베딩)
* **Vector DB:** `ChromaDB` (로컬 벡터 저장소)
* **Python Libs:** `pypdf`, `langchain-text-splitters`, `sentence-transformers`

### 2. 아키텍처 (RAG Flow)



1.  **(준비) `setup_db.py`:** PDF 문서를 로드(`PyPDFLoader`)하고, `chunk_size=1000` 단위로 분할(`RecursiveCharacterTextSplitter`)합니다.
2.  **(준비) Embedding:** 분할된 텍스트 조각들을 `jhgan/ko-sbert-nli` 임베딩 모델을 태워 벡터로 변환한 뒤, `ChromaDB`에 저장합니다.
3.  **(실행) 사용자 질문:** 사용자가 Streamlit UI에 질문을 입력합니다. (예: "회비 얼마야?")
4.  **(실행) 검색:** `ChromaDB`에서 질문과 가장 유사한 벡터(문서 조각 3개, `k=3`)를 검색(`retriever`)합니다.
5.  **(실행) 생성:** 검색된 문서 조각(Context)과 사용자 질문을 `gemma-2b-it` (GGUF/LLM)에 전달하여, **"오직 Context에 기반해서만"** 답변을 생성하도록 합니다.
6.  **(실행) 답변:** 생성된 답변을 Streamlit UI에 출력합니다.

---

##  주요 기술적 도전 및 해결 과정 (Killed / OOM)

이 프로젝트의 핵심 도전 과제는 **제한된 서버 자원(RAM) 내에서 고성능 LLM을 안정적으로 구동**하는 것이었습니다.

### 1. 문제 상황: `Killed` (메모리 부족)

`transformers` 라이브러리를 사용하여 `google/gemma-2b-it` 원본 모델(약 5GB)을 로드하려고 시도했을 때, 모델을 RAM에 불러오는 과정에서 서버의 메모리 한계(OOM)를 초과하여 프로세스가 OS에 의해 강제 종료(`Killed`)되는 문제가 발생했습니다.

### 2. 해결 전략: GGUF 경량화 모델 및 `llama-cpp-python` 도입

단순히 `transformers`를 사용하는 방식으로는 로컬 서빙이 불가능하다고 판단, LLM 서빙 방식을 전면 교체했습니다.

1.  **GGUF (Quantization):**
    `gemma-2b-it` 모델을 4비트로 경량화(Quantization)한 `Q4_K_M.gguf` 버전(약 1.47GB)을 채택했습니다. 이를 통해 모델의 RAM 점유율을 70% 이상 획기적으로 낮췄습니다.

2.  **`llama-cpp-python`:**
    `transformers` 대신 GGUF 모델 구동에 최적화된 C++ 기반 라이브러리 `llama-cpp-python`을 `LangChain`에 연동했습니다. 이는 `transformers` 대비 훨씬 적은 메모리를 사용하면서도 효율적인 추론 속도를 보장했습니다.

### 3. 결과

GGUF와 `llama-cpp-python`의 도입으로, **4GB~8GB 수준의 저사양 VM/서버에서도** 2B급 LLM을 안정적으로 구동하는 데 성공, `Killed` 문제를 완벽하게 해결하고 100% 로컬 챗봇을 완성했습니다.

---

## ⚙️ 설치 및 실행 방법

1.  **리포지토리 클론**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **가상 환경 생성 및 활성화**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **필수 라이브러리 설치 (`requirements.txt`)**
    ```bash
    pip install -r requirements.txt
    ```
    *(requirements.txt)*
    ```
    streamlit
    langchain
    langchain-community
    langchain-huggingface
    langchain-chroma
    llama-cpp-python
    pypdf
    langchain-text-splitters
    sentence-transformers
    ```

4.  **GGUF 모델 다운로드**
    * [Hugging Face](https://huggingface.co/ggml-community/gemma-2b-it-GGUF/tree/main)에서 `gemma-2b-it.Q4_K_M.gguf` (1.47GB) 파일을 다운로드합니다.
    * 다운로드한 `.gguf` 파일을 이 프로젝트 폴더 최상위(`app.py`와 같은 위치)에 복사합니다.

5.  **벡터 DB 생성**
    * `동아리_내규.pdf` 파일을 프로젝트 폴더에 위치시킵니다.
    * `setup_db.py`를 실행하여 `final_vector_db` 폴더를 생성합니다.
    ```bash
    python setup_db.py
    ```

6.  **챗봇 실행**
    ```bash
    streamlit run app.py
    ```
    * 터미널에 `[로컬 GGUF 모델 로드 완료]` 메시지가 뜨면, 웹 브라우저에서 `http://localhost:8501`에 접속합니다.
