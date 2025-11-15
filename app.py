import os
import traceback
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- 1. RAG 체인(Chain) 설정 ---

# (1) LLM 모델 로드 (Hugging Face 로컬 파이프라인)

@st.cache_resource # Streamlit의 캐시 기능을 사용해 모델을 한 번만 로드합니다.
def get_llm_pipeline():
    print("--- [로컬 모델 로드 시작] ---")
    model_id = "google/gemma-2b-it"
    
    # 1. 토크나이저와 모델을 로컬에 다운로드합니다. (최초 실행 시 몇 분 소요)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # 2. Transformers 파이프라인 생성
    pipe = pipeline(
        "text-generation", # (Phi-3와 동일한 타입 사용)
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 3. LangChain에서 사용할 수 있도록 래핑(Wrapping)합니다.
    llm = HuggingFacePipeline(pipeline=pipe)
    print("--- [로컬 모델 로드 완료] ---")
    return llm

# llm 변수에 로드된 모델 파이프라인을 할당합니다.
llm = get_llm_pipeline()
# --------------------------------------------------


# (2) 임베딩 모델 로드 (변경 없음 - 이미 로컬입니다)
@st.cache_resource # 임베딩 모델도 캐시 처리
def get_embeddings():
    model_name = "jhgan/ko-sbert-nli"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
embeddings = get_embeddings()


# (3) 벡터 DB 로드 (변경 없음)
persist_directory = './chroma_db' # (이전에 setup_db.py로 생성한 DB)
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# (4. 5. 6. RAG 체인 및 프롬프트 - 변경 없음)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """
Answer the following 'Question' based *only* on the 'Context' provided.
If the information is not in the context, say "I don't have that information."

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | StrOutputParser()
)


st.title("💬 동아리 규정 안내 챗봇 (Local Ver.)")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("동아리방 사용 시간을 알려줘"):
    
    # 1. 사용자 메시지 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- [ 2. (핵심) 디버깅 블록 추가! ] ---
    # 챗봇이 답변하기 전에, retriever가 무엇을 찾는지 수동으로 확인합니다.
    st.subheader("--- [디버깅: 검색기(Retriever) 결과] ---")
    try:
        # (retriever는 파일 상단에서 이미 로드되었음)
        retrieved_docs = retriever.invoke(user_input)
        
        if retrieved_docs:
            st.write(f"✅ {len(retrieved_docs)}개의 문서를 찾았습니다.")
            for i, doc in enumerate(retrieved_docs):
                st.write(f"--- [문서 {i+1}] ---")
                # 찾은 문서의 내용 일부를 표시
                st.write(f"내용: {doc.page_content[:200]}...")
                # 찾은 문서의 출처(metadata)를 표시
                st.write(f"출처: {doc.metadata}")
        else:
            st.error("❌ 'retriever.invoke'가 아무 문서도 반환하지 못했습니다.")
            st.error("이것이 'I don't have...'의 원인입니다.")

    except Exception as e_debug:
        st.error(f"❌ 'retriever.invoke' 호출 중 치명적 오류 발생: {e_debug}")
    st.write("--- [디버깅 종료] ---")
    # -----------------------------------------------

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성 중입니다..."):
            try:
                # RAG 체인 실행
                response = rag_chain.invoke({"question": user_input})
                st.markdown(response)
                # 3. 챗봇 메시지 저장
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                # (1) 터미널에 강제로 전체 오류 내용(Traceback)을 인쇄합니다.
                print("!!! 치명적인 오류 발생 !!!")
                traceback.print_exc() 
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                
                # (2) 웹 UI에도 오류 메시지 'e'의 내용을 포함하여 표시합니다.
                st.error(f"답변 생성 중 오류가 발생했습니다: {e}")