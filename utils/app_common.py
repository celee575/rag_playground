"""
여러 Streamlit 페이지(app.py, views/*.py)에서 공유하는 공용 유틸.

- Gemini API Key 사이드바 입력 (세션 전역에서 같은 key로 공유됨)
- genai.Client / Chroma PersistentClient 캐싱
- Gemini API 429(RESOURCE_EXHAUSTED)/503(서버 과부하) 재시도 유틸
"""

import random
import time
from pathlib import Path
from typing import Callable

import streamlit as st
from google import genai
from google.genai import errors as genai_errors

from utils.db_utils import ensure_ltm_vector_collection


def get_gemini_api_key() -> str:
    """사이드바에서 Gemini API Key를 입력받는다.

    key="gemini_api_key" 로 고정되어 있어, 어떤 페이지에서 입력하든
    다른 페이지로 이동해도 st.session_state["gemini_api_key"] 값이 유지된다.
    """
    with st.sidebar:
        st.header("설정")
        api_key = st.text_input(
            "Gemini API Key", type="password", key="gemini_api_key"
        )
        st.caption("입력한 키는 세션에서만 사용되며 저장되지 않습니다.")

    if not api_key:
        st.info("좌측 사이드바에 Gemini API Key를 입력하면 이 페이지를 사용할 수 있습니다.")
        st.stop()

    return api_key


@st.cache_resource(show_spinner=False)
def get_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_ltm_collection(chroma_path: str):
    # PersistentClient 기반 컬렉션 로드 (앱 전체에서 1회만 생성되도록 캐싱)
    return ensure_ltm_vector_collection(chroma_path=Path(chroma_path))


# -----------------------------------------------------------------------------
# 재시도 유틸: 429(RESOURCE_EXHAUSTED, 무료 tier 호출 초과), 503(서버 과부하) 대응
# google-genai SDK는 4xx를 ClientError, 5xx를 ServerError로 던지며
# 둘 다 APIError를 상속하고 .code 속성에 HTTP 상태 코드를 담고 있음
# -----------------------------------------------------------------------------
RETRYABLE_STATUS_CODES = {429, 503}


def call_with_retry(
    func: Callable,
    *args,
    max_retries: int = 5,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    on_retry: Callable[[int, int, str, float], None] | None = None,
    **kwargs,
):
    """
    Gemini API 호출을 감싸서 429/503 발생 시 지수 백오프 + 지터로 재시도합니다.
    그 외 에러(400, 403, 404 등)는 재시도 없이 즉시 위로 올립니다.

    on_retry(attempt, max_retries, reason, delay) 콜백을 넘기면
    Streamlit UI(st.toast 등)로 재시도 상황을 알릴 수 있습니다.
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

            delay = min(base_delay * (2**attempt), max_delay) + random.uniform(0, 1)
            reason = "호출 횟수 초과(429)" if status_code == 429 else "서버 과부하(503)"
            if on_retry:
                on_retry(attempt + 1, max_retries, reason, delay)
            time.sleep(delay)

    raise RuntimeError(
        f"최대 재시도 횟수({max_retries}회)를 초과했습니다. 마지막 에러: {last_error}"
    ) from last_error