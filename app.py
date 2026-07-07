"""
Streamlit + Gemini API + ChromaDB(PersistentClient) 기반 RAG 챗봇

- Gemini API Key: 사이드바에서 사용자가 직접 입력 (session_state에만 보관)
- PROMOTION_MODEL / CHATBOT_MODEL / EMBEDDING_MODEL / CSV_PATH / CHROMA_STORE_PATH: st.secrets 로 전달
- LTM(장기기억): ChromaDB semantic search
- 주유소 지역별 평균판매가격: Function Calling(Tool use) 으로 필요할 때만 조회
"""

import json
from pathlib import Path

import chromadb
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

from utils.db_utils import ensure_ltm_vector_collection
from utils.chat_utils import chroma_hits, format_ltm_hit_for_chatbot

# -----------------------------------------------------------------------------
# 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LTM 기반 Gemini 챗봇", page_icon="⛽", layout="wide")
st.title("⛽ LTM + Gemini 챗봇 (주유소 가격 Tool Use)")


# -----------------------------------------------------------------------------
# secrets 로드 (필수 설정값)
# -----------------------------------------------------------------------------
def load_required_secrets() -> dict:
    required_keys = [
        "PROMOTION_MODEL",
        "CHATBOT_MODEL",
        "EMBEDDING_MODEL",
        "CSV_PATH",
        "CHROMA_STORE_PATH",
    ]
    missing = [k for k in required_keys if k not in st.secrets]
    if missing:
        st.error(
            "다음 값이 st.secrets(.streamlit/secrets.toml)에 설정되어 있지 않습니다: "
            f"{', '.join(missing)}"
        )
        st.stop()
    return {k: st.secrets[k] for k in required_keys}


SECRETS = load_required_secrets()
PROMOTION_MODEL = SECRETS["PROMOTION_MODEL"]
CHATBOT_MODEL = SECRETS["CHATBOT_MODEL"]
EMBEDDING_MODEL = SECRETS["EMBEDDING_MODEL"]
CSV_PATH = SECRETS["CSV_PATH"]
CHROMA_STORE_PATH = SECRETS["CHROMA_STORE_PATH"]


# -----------------------------------------------------------------------------
# 사이드바: Gemini API Key 입력
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    st.caption("입력한 키는 세션에서만 사용되며 저장되지 않습니다.")

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

if not gemini_api_key:
    st.info("좌측 사이드바에 Gemini API Key를 입력하면 챗봇을 사용할 수 있습니다.")
    st.stop()


# -----------------------------------------------------------------------------
# Gemini client (API Key 입력 시마다 생성)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


client = get_genai_client(gemini_api_key)


# -----------------------------------------------------------------------------
# ChromaDB PersistentClient + LTM 컬렉션
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ltm_collection(chroma_path: str):
    # PersistentClient 로 디스크에 영속화된 컬렉션 로드
    return ensure_ltm_vector_collection(chroma_path=Path(chroma_path))


ltm_collection = get_ltm_collection(CHROMA_STORE_PATH)


# -----------------------------------------------------------------------------
# 주유소 CSV 로드 (앱 시작 시 1회, 캐시)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_price_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="cp949")
    df = df.set_index("구분")
    df.index = pd.to_datetime(df.index)
    return df


df = load_price_df(CSV_PATH)
REGIONS = df.columns.tolist()


def select_oil_price(start_date: str, end_date: str, regions: list[str] | None = None) -> dict:
    mask = (df.index >= start_date) & (df.index <= end_date)
    result = df.loc[mask]
    if regions:
        result = result[regions]
    return {str(idx.date()): row.dropna().to_dict() for idx, row in result.iterrows()}


select_oil_price_declaration = {
    "name": "select_oil_price",
    "description": (
        "날짜 범위와 지역을 지정해 주유소 일별 평균 판매가격(원/리터)을 조회합니다. "
        "가격 수치가 필요한 질문에만 호출하세요."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "조회 시작일 (YYYY-MM-DD 형식)",
            },
            "end_date": {
                "type": "string",
                "description": "조회 종료일 (YYYY-MM-DD 형식)",
            },
            "regions": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"조회할 지역 목록. 생략 시 전체 조회. 가능한 값: {REGIONS}",
            },
        },
        "required": ["start_date", "end_date"],
    },
}

system_instruction = """
당신은 학습 메모리를 활용하는 한국어 챗봇입니다.
사용자 질문으로 semantic search한 LTM 메모리를 근거로 답하세요.
사용자의 이전 질문 패턴을 반영해 답변하세요.
주유소 가격 등 수치 데이터가 필요한 경우에만 select_oil_price 툴을 호출하세요.
툴 없이 답할 수 있으면 호출하지 마세요.
"""

tools = types.Tool(function_declarations=[select_oil_price_declaration])
generation_config = types.GenerateContentConfig(
    tools=[tools],
    system_instruction=system_instruction,
)


# -----------------------------------------------------------------------------
# 임베딩 + LTM 검색
# -----------------------------------------------------------------------------
def embed_query(text: str) -> list[float]:
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    return [float(v) for v in response.embeddings[0].values]


def build_retrieval_context(question: str) -> dict:
    if ltm_collection.count() == 0:
        return {"query": question, "ltm_context": []}

    query_embedding = embed_query(question)
    search_results = ltm_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, ltm_collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return {
        "query": question,
        "ltm_context": [format_ltm_hit_for_chatbot(hit) for hit in chroma_hits(search_results)],
    }


# -----------------------------------------------------------------------------
# Gemini 호출 (Tool 호출 루프 포함)
# -----------------------------------------------------------------------------
def ask(question: str, retrieval_context: dict) -> str:
    user_prompt = f"""
Semantic search 메모리:
{json.dumps(retrieval_context, ensure_ascii=False, indent=2)}

사용자 질문: {question}
"""
    contents = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

    while True:
        response = client.models.generate_content(
            model=CHATBOT_MODEL,
            contents=contents,
            config=generation_config,
        )

        candidate = response.candidates[0].content
        contents.append(candidate)

        tool_calls = [p for p in candidate.parts if p.function_call]
        if not tool_calls:
            return response.text

        tool_results = []
        for part in tool_calls:
            fc = part.function_call
            args = dict(fc.args)

            if fc.name == "select_oil_price":
                result = select_oil_price(**args)
            else:
                result = {"error": f"알 수 없는 함수: {fc.name}"}

            tool_results.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )

        contents.append(types.Content(role="tool", parts=tool_results))


# -----------------------------------------------------------------------------
# Streamlit 채팅 UI
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("질문을 입력하세요 (예: 환율과 지역별 주유소 가격의 공통된 흐름을 요약해줘)")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("LTM 검색 및 답변 생성 중..."):
            try:
                retrieval_context = build_retrieval_context(user_question)
                answer = ask(user_question, retrieval_context)
            except Exception as e:
                answer = f"오류가 발생했습니다: {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})