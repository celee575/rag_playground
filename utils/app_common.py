"""
여러 Streamlit 페이지(app.py, pages/*.py)에서 공유하는 공용 유틸.

- Gemini API Key 사이드바 입력 (세션 전역에서 같은 key로 공유됨)
- genai.Client / Chroma PersistentClient 캐싱
"""

from pathlib import Path

import streamlit as st
from google import genai

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