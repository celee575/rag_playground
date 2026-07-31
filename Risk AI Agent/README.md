# 1. ChromaDB란?
ChromaDB는 **오픈소스 벡터 데이터베이스(Vector DB)**    
주로 LLM 기반 애플리케이션에서 문서의 임베딩을 저장하고 유사도 검색을 위해 사용  
특히 문서 기반 검색, 사내 챗봇, 개인 지식 관리 시스템을 만들 때 유용  

### 주요 특징
- 설치가 간단하고 로컬 사용에 최적화  
- 자체적으로 임베딩 저장, 검색, 필터링 기능 지원  
- LangChain, LlamaIndex와도 쉽게 통합 가능  
- 개발자가 로컬 테스트 용도로 빠르게 사용할 수 있는 "zero-config" 벡터 DB  
<br>

# 2. ChromaDB 설치 방법
Python 3.8 이상 필요
```
pip install chromadb
```
<br>

# 3. 기본 사용 예제 
## 1) 로컬 - PersistentClient()
You can configure Chroma to save and load the database from your local machine, using the `PersistentClient()`.  
```
import chromadb
from chromadb.config import Settings

# ChromaDB 클라이언트 생성
client = chromadb.PersistentClient(
    path="./chroma_db" , # 로컬 저장 경로
    settings=Settings(anonymized_telemetry=False), # chromadb 통계 데이터 사용 비활성화
)

# 컬렉션 생성
collection = client.get_or_create_collection(name="my_collection")

# 문서 추가
collection.add(
    ids=["id1", "id2"], # 고유 문서 번호 id 필수
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ]
)

# 검색 (쿼리와 가장 유사한 문서 반환)
results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print(results)
```
<br>

## 2) 서버 방식 - Client-Server Mode

chromadb 서버 실행 명령어
```
chroma run --path /db_path
```  

### 1) Sync - HttpClient()
you can deploy single-node Chroma to a Docker container, or a machine hosted by a cloud provider like AWS, GCP, Azure, and others.  
```
import chromadb

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
```

### 2) Async - AsyncHttpClient()

```
import asyncio
import chromadb

async def main():
    client = await chromadb.AsyncHttpClient(host='localhost', port=8000)

    collection = await client.create_collection(name="my_collection")
    await collection.add(
        documents=["hello world"],
        ids=["id1"]
    )

asyncio.run(main())
```
<br>

# 4. ChromaDB를 LLM과 함께 사용하는 구조
- RAG (Retrieval-Augmented Generation)  
- 문서 업로드 → 임베딩 생성 → ChromaDB에 저장 → 사용자 질문 임베딩 → 유사 문서 검색 → LLM에 전달  
- 필요한 도구:
    - OpenAI 또는 HuggingFace 임베딩 API
    - LangChain 또는 직접 구현
    - LLM API (GPT-4, Claude 등)
<br>
<br>

# 5. LLM과 함께 사용하는 예제
### 예제 1: 문서 저장 + 검색 + GPT로 답변 생성
```
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import OpenAI 
from langchain.chains import RetrievalQA 

# 1. OpenAI 임베딩 사용 
embedding = OpenAIEmbeddings() 

# 2. Chroma DB 연결 
vectordb = Chroma(persist_directory="./chroma_store", embedding_function=embedding) 

# 3. 문서 저장 (예: 간단한 문장) 
vectordb.add_texts(["GPT는 OpenAI에서 개발한 모델입니다", "한국의 수도는 서울입니다"]) 

# 4. 검색 QA 체인 생성 
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=vectordb.as_retriever()) 

# 5. 질문 → 검색된 내용 기반으로 GPT 응답 
query = "OpenAI에서 만든 모델 이름은?" 
response = qa.run(query) 
print(response)
```
<br>

# 6. 활용 예시
- 🔍 문서 질문 응답
    - 회사 문서 업로드 → 관련 답변 생성
- 📚 PDF 검색 
    - PDF → 텍스트 변환 → ChromaDB 저장 → 유사 검색  
- 🧠 사내 지식베이스 
    - 사내 위키, 매뉴얼을 LLM이 참조 가능하게 구성  
- 💬 채팅봇 + 문서 검색	
    - LLM 기반 챗봇이 ChromaDB에 저장된 자료 참조  

<br>
