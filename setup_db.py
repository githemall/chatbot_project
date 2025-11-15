import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


try:
    loader = PyPDFLoader("/home/asd/projects/chatbot/동아리_내규.pdf")
    docs = loader.load()
    print(f"총 {len(docs)}개의 페이지를 로드했습니다.")
except Exception as e:
    print(f"문서 로드 중 오류: {e}")
    exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"문서를 총 {len(splits)}개의 조각(chunk)으로 분할했습니다.")
# --------------------------------------------------

# 3. 임베딩 및 벡터 DB 저장 (수정된 부분)
# 로컬 DB 폴더 이름을 새로 지정하는 것이 좋습니다.
persist_directory = './chroma_db'

try:
 
    model_name = "jhgan/ko-sbert-nli" 
    model_kwargs = {'device': 'cuda'} 
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("로컬 임베딩 모델을 성공적으로 로드했습니다.")
    # --------------------------------------------------

    # Chroma DB에 저장
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"벡터 DB가 '{persist_directory}' 폴더에 성공적으로 저장되었습니다.")

except Exception as e:
    print(f"임베딩 또는 DB 저장 중 오류 발생: {e}")