import time
import random
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from utils.db_utils import ensure_ltm_vector_collection
from utils.chat_utils import chroma_hits, format_ltm_hit_for_chatbot

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

CHROMA_STORE_PATH = PROJECT_ROOT / "data" / "chroma_db"
PROMOTION_MODEL = "gemini-3.5-flash"
CHATBOT_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

##--------------반드시 입력해주세요!--------------------
GEMINI_API_KEY = ""
##------------------------------------------------------

if not GEMINI_API_KEY:
    raise RuntimeError(".env에 GEMINI_API_KEY 또는 GOOGLE_API_KEY를 설정하세요.")

client = genai.Client(api_key=GEMINI_API_KEY)

# chromadb client 생성 (뉴스 기사 LTM 컬렉션)
ltm_collection = ensure_ltm_vector_collection(chroma_path=CHROMA_STORE_PATH)


# ──────────────────────────────────────────────────────────────────────────
# 재시도 유틸: 429(RESOURCE_EXHAUSTED, 무료 tier 호출 초과), 503(서버 과부하) 대응
# google-genai SDK는 4xx를 ClientError, 5xx를 ServerError로 던지며
# 둘 다 APIError를 상속하고 .code 속성에 HTTP 상태 코드를 담고 있음
# ──────────────────────────────────────────────────────────────────────────
RETRYABLE_STATUS_CODES = {429, 503}


def call_with_retry(
    func,
    *args,
    max_retries: int = 5,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    **kwargs,
):
    """
    Gemini API 호출을 감싸서 429/503 발생 시 지수 백오프 + 지터로 재시도합니다.
    그 외 에러(400, 403, 404 등)는 재시도 없이 즉시 위로 올립니다.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except genai_errors.APIError as e:
            status_code = getattr(e, "code", None)
            if status_code not in RETRYABLE_STATUS_CODES:
                raise  # 재시도 대상이 아닌 에러는 바로 전파

            last_error = e
            if attempt == max_retries - 1:
                break

            delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 1)
            reason = "호출 횟수 초과(429)" if status_code == 429 else "gemini 서버 과부하(503)"
            print(
                f"[재시도 {attempt + 1}/{max_retries}] {reason} - "
                f"{delay:.1f}초 대기 후 재시도합니다."
            )
            time.sleep(delay)

    raise RuntimeError(
        f"최대 재시도 횟수({max_retries}회)를 초과했습니다. 마지막 에러: {last_error}"
    ) from last_error


def compact_embedding(text: str) -> list[float]:
    def _embed():
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
        return response.embeddings[0].values

    embedding = call_with_retry(_embed)
    return [float(value) for value in embedding]


# ──────────────────────────────────────────────────────────────────────────
# Tool 1: news_search — 뉴스 기사 semantic search (LTM 벡터 검색)
# 일자 인식은 Gemini의 파라미터 추출 자체에 맡김 (별도 전처리 단계 없음)
# ──────────────────────────────────────────────────────────────────────────
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

    where_filter = None
    if start_date or end_date:
        conditions = []
        if start_date:
            conditions.append({"pubDate": {"$gte": start_date}})
        if end_date:
            conditions.append({"pubDate": {"$lte": end_date}})
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


# ──────────────────────────────────────────────────────────────────────────
# Tool 2: select_oil_price — 지역별 주유소 평균판매가격 조회 (기존 로직 그대로)
# ──────────────────────────────────────────────────────────────────────────
df = pd.read_csv("주유소_지역별_평균판매가격.csv", encoding="cp949")
df = df.set_index("구분")
df.index = pd.to_datetime(df.index)

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


def select_oil_price(
    start_date: str, end_date: str, regions: list[str] | None = None
) -> dict:
    mask = (df.index >= start_date) & (df.index <= end_date)
    result = df.loc[mask]

    if regions:
        result = result[regions]

    return {
        str(idx.date()): row.dropna().to_dict() for idx, row in result.iterrows()
    }


# ──────────────────────────────────────────────────────────────────────────
# 두 tool을 모두 등록 — 어떤 tool을 호출할지 판단은 Gemini에게 맡김
# ──────────────────────────────────────────────────────────────────────────
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

tools = types.Tool(
    function_declarations=[news_search_declaration, select_oil_price_declaration]
)
config = types.GenerateContentConfig(
    tools=[tools],
    system_instruction=system_instruction,
)


# ──────────────────────────────────────────────────────────────────────────
# Tool 호출 처리 루프 (재시도 로직 포함)
# ──────────────────────────────────────────────────────────────────────────
def ask(question: str) -> str:
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    while True:
        response = call_with_retry(
            client.models.generate_content,
            model=CHATBOT_MODEL,
            contents=contents,
            config=config,
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
                    # tool 내부에서 compact_embedding 등 Gemini 호출이 재시도를
                    # 모두 소진하고도 실패한 경우 여기로 전파됨 -> 상위로 재전파
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


if __name__ == "__main__":
    # question = "5월 미국과 이란 관계를 볼 때 6월 주유소 가격을 예측해주세요"
    # question = "5월 주유소 지역별 금액을 분석해주세요"
    # question = "최근 뉴스 기사 요약해주세요"
    # question = "안녕하세요"
    question = "최근 금리 인상이 유가에 미친 영향에 대한 기사가 있나요?"
    print(ask(question))