"""
Streamlit + Gemini API + ChromaDB(PersistentClient) 기반 RAG 챗봇 (챗봇 페이지)

- Gemini API Key: 사이드바에서 사용자가 직접 입력 (utils.app_common 공용 함수)
- PROMOTION_MODEL / CHATBOT_MODEL / EMBEDDING_MODEL / CSV_PATH / CHROMA_STORE_PATH: st.secrets 로 전달
- Tool 1: news_search      -> 뉴스 기사 semantic search (LTM 벡터 검색)
- Tool 2: select_oil_price -> 지역별 주유소 평균판매가격 조회
- 두 tool 모두 Gemini에게 등록해두고, 어떤 tool을 호출할지는 Gemini가 판단
- 429(RESOURCE_EXHAUSTED)/503(서버 과부하) 재시도: utils.app_common.call_with_retry
"""

import pandas as pd
import streamlit as st
from google.genai import types
from google.genai import errors as genai_errors

from utils.app_common import (
    call_with_retry,
    get_gemini_api_key,
    get_genai_client,
    get_ltm_collection,
)
from utils.chat_utils import chroma_hits, format_ltm_hit_for_chatbot

st.title("⛽ LTM + Gemini 챗봇 (뉴스 검색 + 주유소 가격 Tool Use)")


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
# 사이드바: Gemini API Key 입력 (다른 페이지와 세션 공유)
# -----------------------------------------------------------------------------
gemini_api_key = get_gemini_api_key()

with st.sidebar:
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Gemini client / ChromaDB PersistentClient + LTM 컬렉션 (캐싱, 페이지 간 공유)
# -----------------------------------------------------------------------------
client = get_genai_client(gemini_api_key)
ltm_collection = get_ltm_collection(CHROMA_STORE_PATH)


# -----------------------------------------------------------------------------
# 임베딩 (재시도 로직 포함)
# -----------------------------------------------------------------------------
def compact_embedding(text: str) -> list[float]:
    def _embed():
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return response.embeddings[0].values

    embedding = call_with_retry(
        _embed,
        on_retry=lambda attempt, max_retries, reason, delay: st.toast(
            f"{reason} - {delay:.1f}초 후 재시도 ({attempt}/{max_retries})"
        ),
    )
    return [float(value) for value in embedding]


# -----------------------------------------------------------------------------
# Tool 1: news_search — 뉴스 기사 semantic search (LTM 벡터 검색)
# -----------------------------------------------------------------------------
news_search_declaration = {
    "name": "news_search",
    "description": (
        "뉴스 기사에 대한 의미 기반(semantic) 유사도 검색을 수행합니다. "
        "최근 동향, 특정 사건, 국제 정세 등 뉴스 맥락이 필요한 질문에만 호출하세요. "
        "질문에 특정 일자나 기간이 언급되면 start_date/end_date에 인식한 날짜를 채워 넣으세요."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색할 질문 또는 핵심 키워드",
            },
            "start_date": {
                "type": "string",
                "description": "검색 대상 시작일 (YYYY-MM-DD). 언급 없으면 생략.",
            },
            "end_date": {
                "type": "string",
                "description": "검색 대상 종료일 (YYYY-MM-DD). 언급 없으면 생략.",
            },
        },
        "required": ["query"],
    },
}


def news_search(
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    n_results: int = 3,
) -> dict:
    query_embedding = compact_embedding(query)

    # 뉴스 메타데이터에 'date' 필드(YYYY-MM-DD 문자열)가 있다고 가정.
    # chromadb 버전에 따라 문자열 범위 비교($gte/$lte) 지원 여부가 다르니
    # 실제 컬렉션의 메타데이터 스키마와 chromadb 버전을 확인하세요.
    where_filter = None
    if start_date or end_date:
        conditions = []
        if start_date:
            conditions.append({"date": {"$gte": start_date}})
        if end_date:
            conditions.append({"date": {"$lte": end_date}})
        where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    if not ltm_collection.count():
        return {"query": query, "hits": []}

    results = ltm_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, max(1, ltm_collection.count())),
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )

    hits = chroma_hits(results)
    return {
        "query": query,
        "hits": [format_ltm_hit_for_chatbot(hit) for hit in hits],
    }


# -----------------------------------------------------------------------------
# Tool 2: select_oil_price — 지역별 주유소 평균판매가격 조회
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_price_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="cp949")
    df = df.set_index("구분")
    df.index = pd.to_datetime(df.index)
    return df


df = load_price_df(CSV_PATH)
REGIONS = df.columns.tolist()

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


def select_oil_price(start_date: str, end_date: str, regions: list[str] | None = None) -> dict:
    mask = (df.index >= start_date) & (df.index <= end_date)
    result = df.loc[mask]

    if regions:
        result = result[regions]

    return {str(idx.date()): row.dropna().to_dict() for idx, row in result.iterrows()}


# -----------------------------------------------------------------------------
# 두 tool을 모두 등록 — 어떤 tool을 호출할지 판단은 Gemini에게 맡김
# -----------------------------------------------------------------------------
TOOL_REGISTRY = {
    "news_search": news_search,
    "select_oil_price": select_oil_price,
}

system_instruction = """
당신은 뉴스와 유가 데이터를 분석하는 한국어 챗봇입니다.
사용자 질문에 따라 아래 두 tool 중 필요한 것만 호출하세요.

- news_search: 최근 동향, 사건, 국제 정세 등 뉴스 맥락이 필요할 때
- select_oil_price: 지역별 유가 수치가 필요할 때

두 정보가 모두 필요한 복합 질문이면 두 tool을 모두 호출한 뒤 결과를 종합해 답변하세요.
어느 tool도 필요 없는 일반적인 질문이면 tool을 호출하지 말고,
당신의 역할(뉴스·유가 데이터 분석)에 맞는 질문을 하도록 자연스럽게 안내하세요.
답변에는 어떤 정보를 근거로 답했는지(뉴스 검색 결과인지, 유가 데이터인지) 드러나게 작성하세요.
"""

tools = types.Tool(function_declarations=[news_search_declaration, select_oil_price_declaration])
generation_config = types.GenerateContentConfig(
    tools=[tools],
    system_instruction=system_instruction,
)


# -----------------------------------------------------------------------------
# Tool 호출 처리 루프 (재시도 로직 포함)
# -----------------------------------------------------------------------------
def ask(question: str) -> str:
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    while True:
        response = call_with_retry(
            client.models.generate_content,
            model=CHATBOT_MODEL,
            contents=contents,
            config=generation_config,
            on_retry=lambda attempt, max_retries, reason, delay: st.toast(
                f"{reason} - {delay:.1f}초 후 재시도 ({attempt}/{max_retries})"
            ),
        )

        candidate = response.candidates[0].content
        contents.append(candidate)

        tool_calls = [p for p in candidate.parts if p.function_call]
        if not tool_calls:
            return response.text  # 최종 답변

        tool_results = []
        for part in tool_calls:
            fc = part.function_call
            args = dict(fc.args)
            func = TOOL_REGISTRY.get(fc.name)

            if func is None:
                result = {"error": f"알 수 없는 함수: {fc.name}"}
            else:
                try:
                    result = func(**args)
                except genai_errors.APIError:
                    # tool 내부(예: news_search -> compact_embedding)에서
                    # 재시도를 모두 소진하고도 실패한 경우 -> 상위로 재전파
                    raise
                except Exception as e:
                    result = {"error": f"{fc.name} 실행 중 오류: {e}"}

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

user_question = st.chat_input(
    "질문을 입력하세요 (예: 중동 정세와 국내 유가에 어떤 관련이 있어?)"
)

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Tool 판단 및 답변 생성 중..."):
            try:
                answer = ask(user_question)
            except Exception as e:
                answer = f"오류가 발생했습니다: {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})