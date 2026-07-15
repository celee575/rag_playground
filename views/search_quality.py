"""
LTM(ChromaDB) 검색 품질 검증 페이지

- Gemini API Key: 사이드바 입력 (utils.app_common 공용 함수, 다른 페이지와 세션 공유)
- EMBEDDING_MODEL, CHROMA_STORE_PATH: st.secrets 로 전달
- query_text: 사용자가 직접 입력 (예: "국제유가가 하락한 이유는?")
"""

import pandas as pd
import streamlit as st

from utils.app_common import (
    call_with_retry,
    get_gemini_api_key,
    get_genai_client,
    get_ltm_collection,
)

st.title("🔍 검색 품질 검증")
st.caption("입력한 질문이 ChromaDB에서 어떤 문서를 얼마나 가깝게 검색해오는지 확인합니다.")


# -----------------------------------------------------------------------------
# secrets
# -----------------------------------------------------------------------------
def load_required_secrets() -> dict:
    required_keys = ["EMBEDDING_MODEL", "CHROMA_STORE_PATH"]
    missing = [k for k in required_keys if k not in st.secrets]
    if missing:
        st.error(
            "다음 값이 st.secrets(.streamlit/secrets.toml)에 설정되어 있지 않습니다: "
            f"{', '.join(missing)}"
        )
        st.stop()
    return {k: st.secrets[k] for k in required_keys}


SECRETS = load_required_secrets()
EMBEDDING_MODEL = SECRETS["EMBEDDING_MODEL"]
CHROMA_STORE_PATH = SECRETS["CHROMA_STORE_PATH"]


# -----------------------------------------------------------------------------
# API Key / client / collection (app.py와 동일한 공용 함수 사용 -> 세션 공유)
# -----------------------------------------------------------------------------
gemini_api_key = get_gemini_api_key()
client = get_genai_client(gemini_api_key)
ltm_collection = get_ltm_collection(CHROMA_STORE_PATH)


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
# 컬렉션 현황
# -----------------------------------------------------------------------------
total_docs = ltm_collection.count()
st.metric("문서 수", total_docs)

if total_docs == 0:
    st.warning("현재 저장된 문서가 없습니다. 검색 결과가 항상 비어있습니다.")


# -----------------------------------------------------------------------------
# 질문 입력 + 검색 실행
# -----------------------------------------------------------------------------
example_questions = [
    "국제유가가 하락한 이유는?",
    "중동 전쟁이 국내 유가에 어떤 영향을 주고 있나?",
]

with st.form("search_quality_form"):
    query_text = st.text_input(
        "검증할 질문을 입력하세요",
        placeholder=" / ".join(example_questions),
    )
    n_results_input = st.slider("조회할 결과 개수(n_results)", min_value=1, max_value=10, value=3)
    submitted = st.form_submit_button("검색 실행")

if not submitted:
    st.stop()

if not query_text.strip():
    st.warning("질문을 입력해 주세요.")
    st.stop()

if total_docs == 0:
    st.stop()

with st.spinner("임베딩 생성 및 Chroma 검색 중..."):
    query_embedding = compact_embedding(query_text)

    n_results = min(n_results_input, total_docs)
    ltm_search_results = ltm_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

documents = ltm_search_results.get("documents", [[]])[0]
metadatas = ltm_search_results.get("metadatas", [[]])[0]
distances = ltm_search_results.get("distances", [[]])[0]
ids = ltm_search_results.get("ids", [[None] * len(documents)])[0] if "ids" in ltm_search_results else [None] * len(documents)

hit_count = len(documents)

st.subheader("검증 요약")
col1, col2 = st.columns(2)
col1.metric("검색된 히트 수", hit_count)
if distances:
    col2.metric("최상위(1위) distance", f"{distances[0]:.4f}")

st.caption(
    "distance는 값이 작을수록 질문과 더 가깝게(유사하게) 매칭되었다는 뜻입니다. "
)

if hit_count == 0:
    st.info("검색된 문서가 없습니다.")
    st.stop()


# -----------------------------------------------------------------------------
# 결과 테이블
# -----------------------------------------------------------------------------
st.subheader("검색 결과")

rows = []
for rank, (doc, meta, dist, doc_id) in enumerate(zip(documents, metadatas, distances, ids), start=1):
    preview = doc if len(doc) <= 200 else doc[:200] + "..."
    rows.append(
        {
            "순위": rank,
            "distance": round(float(dist), 4),
            "id": doc_id,
            "문서 미리보기": preview,
            "metadata": meta,
        }
    )

result_df = pd.DataFrame(rows)
st.dataframe(
    result_df[["순위", "distance", "id", "문서 미리보기"]],
    width='stretch',
    hide_index=True,
)

st.bar_chart(result_df.set_index("순위")["distance"])

st.subheader("문서별 상세 보기")
for row in rows:
    with st.expander(f"{row['순위']}위 · distance={row['distance']} · id={row['id']}"):
        st.markdown("**문서 전문**")
        full_doc = documents[row["순위"] - 1]
        st.write(full_doc)
        st.markdown("**metadata**")
        st.json(row["metadata"])


# -----------------------------------------------------------------------------
# 원본 응답 (raw) 확인용
# -----------------------------------------------------------------------------
with st.expander("Chroma 원본 응답(raw) 보기"):
    st.json(
        {
            "query_text": query_text,
            "ltm_search_hit_count": hit_count,
            "ltm_search_results": ltm_search_results,
        }
    )