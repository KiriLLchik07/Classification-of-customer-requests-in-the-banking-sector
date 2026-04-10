from pathlib import Path
import streamlit as st
from backend_client import DEFAULT_BACKEND_URL, get_model_info, get_models
from styles.load_css import STYLES_PATH, load_css

load_css(Path(STYLES_PATH) / "models_info_page.css")

backend_url = st.session_state.get("backend_url", DEFAULT_BACKEND_URL)
models_result = get_models(backend_url)
fallback_models = ["Logistic Regression", "GRU", "DistilBERT"]
model_names = models_result["models"] if models_result["ok"] and models_result["models"] else fallback_models

st.title("Models info", text_alignment="center")

st.markdown(
    f"""
    <br>
    <div class="info-box">
        On this page, you can learn more about the algorithms used in our system and
        which are used to predict the category of a client's request.
        <br><br>
        Available models from backend:
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

model_descriptions = {
    "Logistic Regression": (
        "Baseline linear classifier used with TF-IDF features. Fast and interpretable."
    ),
    "GRU": (
        "Recurrent neural architecture that processes text as token sequences."
    ),
    "DistilBERT": (
        "Transformer architecture with the highest quality in current experiments."
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

model_info_result = get_model_info(backend_url, selected_model)
if model_info_result["ok"]:
    versions = model_info_result["versions"]
    if versions:
        st.subheader("Registry versions")
        st.dataframe(versions, use_container_width=True, hide_index=True)
    else:
        st.info("Model is available, but no versions were returned.")
else:
    st.warning(f"Could not fetch registry info: {model_info_result['error']}")
