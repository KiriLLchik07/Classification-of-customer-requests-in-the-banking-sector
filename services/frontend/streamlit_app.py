import streamlit as st
from pathlib import Path
from backend_client import DEFAULT_BACKEND_URL, get_health, normalize_backend_url
from styles.load_css import STYLES_PATH, load_css

st.set_page_config(
    page_title="Banking Request Classification",
    page_icon=":bank:",
    layout="wide",
)

load_css(Path(STYLES_PATH) / "common.css")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL

with st.sidebar:
    st.markdown("<div class='sidebar-title'>Navigation bar</div>", unsafe_allow_html=True)
    backend_url_input = st.session_state.backend_url

    normalized_url = normalize_backend_url(backend_url_input) or DEFAULT_BACKEND_URL
    st.session_state.backend_url = normalized_url

    health_result = get_health(normalized_url)
    if health_result["ok"]:
        st.markdown(
            "<div class='sidebar-status sidebar-status-ok'>System is working</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='sidebar-status sidebar-status-warning'>Backend unavailable</div>",
            unsafe_allow_html=True,
        )

page_1 = st.Page("pages/home_page.py", title="Home page")
page_2 = st.Page("pages/prediction_page.py", title="Predict request type")
page_3 = st.Page("pages/models_info_page.py", title="Models info")

pg = st.navigation([page_1, page_2, page_3])

pg.run()
