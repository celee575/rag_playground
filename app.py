"""
앱 진입점(라우터).

실제 페이지 내용은 views/ 폴더에 있고, 여기서는
사이드바에 표시될 페이지 이름/아이콘/순서만 정의한다.
(파일명과 무관하게 title을 직접 지정할 수 있음)
"""

import streamlit as st

st.set_page_config(page_title="RAG 기반 Gemini 챗봇", page_icon="⛽", layout="wide")

chat_page = st.Page("views/chat.py", title="RAG 기반 Gemini 챗봇", icon="⛽", default=True)
search_quality_page = st.Page("views/search_quality.py", title="검색 품질 검증", icon="🔍")

pg = st.navigation([chat_page, search_quality_page])
pg.run()