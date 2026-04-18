from pathlib import Path
import streamlit as st
from backend_client import DEFAULT_BACKEND_URL, get_model_info, get_mlflow_models
from styles.load_css import STYLES_PATH, load_css

load_css(Path(STYLES_PATH) / "models_info_page.css")

backend_url = st.session_state.get("backend_url", DEFAULT_BACKEND_URL)
models_result = get_mlflow_models(backend_url)
fallback_models = [
    "Banking77_LogisticRegression",
    "Banking77_GRU",
    "Banking77_LSTM",
    "Banking77_BERT",
    "Banking77_DistilBERT",
]
model_names = (
    models_result["model_names"]
    if models_result["ok"] and models_result["model_names"]
    else fallback_models
)

st.title("Models info", text_alignment="center")

st.markdown(
    f"""
    <br>
    <div class="info-box">
        On this page, you can learn more about the models used in the system.
        <br><br>
        Models from MLflow Registry:
        <ul>
            {''.join(f'<li>{name}</li>' for name in model_names)}
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

if not models_result["ok"]:
    st.warning(f"Could not fetch model list: {models_result['error']}")

selected_model = st.radio(
    "Model tabs",
    model_names,
    horizontal=True,
    label_visibility="collapsed",
    key="model_info_radio",
)

selected_alias = st.selectbox(
    "Model alias",
    ["production", "reserve", "baseline"],
    index=0,
    key="model_info_alias_select",
    label_visibility="collapsed",
)

model_descriptions = {
    "Banking77_LogisticRegression": (
        "Baseline linear classifier with TF-IDF features. Fast inference and strong interpretability."
    ),
    "Banking77_GRU": (
        "Recurrent neural model (GRU) that captures sequential text patterns with moderate latency."
    ),
    "Banking77_LSTM": (
        "Recurrent neural model (LSTM) designed for sequential dependencies in customer requests."
    ),
    "Banking77_BERT": (
        "Full BERT transformer with high quality and higher computational cost."
    ),
    "Banking77_DistilBERT": (
        "Compact transformer balancing strong quality and low latency for production serving."
    ),
}

description = model_descriptions.get(
    selected_model,
    "Model description is not specified yet.",
)

st.markdown(
    f"""
    <div class="model-details-box">
        <p class="model-details-title">{selected_model}</p>
        {description}
    </div>
    """,
    unsafe_allow_html=True,
)

model_info_result = get_model_info(
    backend_url=backend_url,
    model_name=selected_model,
    alias=selected_alias,
)
if model_info_result["ok"]:
    versions = model_info_result["versions"]
    if versions:
        st.subheader(f"Registry version for alias: {selected_alias}")
        st.dataframe(versions, use_container_width=True, hide_index=True)
    else:
        st.info("Model is available, but no versions were returned.")
else:
    st.warning(f"Could not fetch registry info: {model_info_result['error']}")
