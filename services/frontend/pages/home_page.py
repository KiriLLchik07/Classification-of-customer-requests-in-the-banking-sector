from pathlib import Path
import streamlit as st
from backend_client import DEFAULT_BACKEND_URL, get_health, get_models
from styles.load_css import STYLES_PATH, load_css

load_css(Path(STYLES_PATH) / "home_page.css")

backend_url = st.session_state.get("backend_url", DEFAULT_BACKEND_URL)
health_result = get_health(backend_url)
models_result = get_models(backend_url)
loaded_models = models_result.get("models", [])

st.title("Classification of client's requests in banking sector", text_alignment="center")

st.markdown(
    f"""
    <br>
    <div class="about-project">
        This project solves the task of classifying client requests in the banking sector.
        The system receives request text, predicts its category, and can be used to route
        the request to the required support scenario.
        <br><br>
        The following models are currently used:
        Logistic Regression, GRU, DistilBERT.
        <br><br>
        The system includes:
        <ul>
            <li>FastAPI backend service</li>
            <li>MLflow tracking and model registry</li>
            <li>Docker Compose based infrastructure</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <br>
    <div class='section-caption'>
        Summary information about the algorithms used in the system:
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

col_lr, col_gru, col_distilbert = st.columns(3, gap="large")

col_lr.markdown(
    """
    <div class="short-info-card">
        <p class="card-title">Logistic Regression</p>
        Baseline model in conjunction with TF-IDF features.
        F1-macro: 0.866
    </div>
    """,
    unsafe_allow_html=True,
)

col_gru.markdown(
    """
    <div class="short-info-card">
        <p class="card-title">GRU</p>
        Recurrent neural model that handles sequence patterns in text.
        F1-macro: 0.875
    </div>
    """,
    unsafe_allow_html=True,
)

col_distilbert.markdown(
    """
    <div class="short-info-card">
        <div class="card-title">DistilBERT</div>
        Transformer model with the best quality on current experiments.
        F1-macro: 0.924
    </div>
    """,
    unsafe_allow_html=True,
)

if loaded_models:
    st.markdown(
        f"<div class='loaded-models'>Loaded models in backend: {', '.join(loaded_models)}</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='loaded-models'>Loaded models in backend: none</div>",
        unsafe_allow_html=True,
    )

left_1, right_1 = st.columns([4, 1.3])
left_1.markdown(
    "<div class='page-text'>More details about models are available on the Models info page.</div>",
    unsafe_allow_html=True,
)
if right_1.button("Models info", key="home_models_info_btn", use_container_width=True):
    st.switch_page("pages/models_info_page.py")

left_2, right_2 = st.columns([4, 1.3])
left_2.markdown(
    "<div class='page-text'>To classify your own request text, open the Prediction page.</div>",
    unsafe_allow_html=True,
)
if right_2.button("Prediction", key="home_prediction_btn", use_container_width=True):
    st.switch_page("pages/prediction_page.py")
