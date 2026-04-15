from pathlib import Path
import time
import streamlit as st
from backend_client import DEFAULT_BACKEND_URL, get_models, predict_request, get_mlflow_models
from styles.load_css import STYLES_PATH, load_css

load_css(Path(STYLES_PATH) / "prediction_page.css")

backend_url = st.session_state.get("backend_url", DEFAULT_BACKEND_URL)
loaded_models = get_models(backend_url)
mlflow_models = get_mlflow_models(backend_url)

fallback_models = ["DistilBERT", "GRU", "Logistic Regression"]
loaded_model_names = loaded_models["models"] if loaded_models["ok"] and loaded_models["models"] else fallback_models
registry_model_names = (
    mlflow_models["model_names"] if mlflow_models["ok"] and mlflow_models["model_names"] else []
)
other_registry_models = [name for name in registry_model_names if name not in loaded_model_names]

st.title("Prediction section", text_alignment="center")

st.markdown(
    """
    <br>
    <div class="info-box">
        Here, the system automatically determines the category of the client's request
        using machine learning algorithms and can redirect it to the appropriate support scenario.
    </div>
    """,
    unsafe_allow_html=True,
)

if not loaded_models["ok"]:
    st.warning(f"Could not load models from backend: {loaded_models['error']}")
if not mlflow_models["ok"]:
    st.warning(f"Could not load model list from MLflow Registry: {mlflow_models['error']}")

request_text = st.text_area(
    "Request text",
    label_visibility="collapsed",
    key="request_text",
)

left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    selected_loaded_model = st.selectbox(
        "Loaded model",
        loaded_model_names,
        index=0,
        key="loaded_model_select",
        label_visibility="collapsed",
    )
    selected_registry_model = st.selectbox(
        "Other models from MLflow Registry",
        ["Not selected"] + other_registry_models,
        index=0,
        key="registry_model_select",
        label_visibility="collapsed",
    )
    selected_alias = st.selectbox(
        "Model alias",
        ["production", "reserve", "baseline"],
        index=0,
        key="model_alias_select",
        label_visibility="collapsed",
    )

effective_model = selected_registry_model if selected_registry_model != "Not selected" else selected_loaded_model

with right_col:
    st.markdown(
        f"""
        <div class="hint-box">
            <span class="hint-model">{effective_model}</span> is currently selected for prediction.
            More details about every model are available on the <b>Models info</b> page.
        </div>
        """,
        unsafe_allow_html=True,
    )

predict_clicked = st.button("Make a prediction", key="predict_btn", use_container_width=False)

if predict_clicked:
    clean_text = request_text.strip()
    if not clean_text:
        st.error("Request text cannot be empty.")
    else:
        start = time.perf_counter()
        prediction_result = predict_request(
            backend_url=backend_url,
            text=clean_text,
            model_name=effective_model,
            model_alias=selected_alias,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if prediction_result["ok"]:
            confidence = prediction_result.get("confidence")
            confidence_block = f"<br>Confidence: <b>{confidence:.3f}</b>" if confidence is not None else ""
            st.markdown(
                f"""
                <div class="result-box">
                    Category of your request: <b>{prediction_result['prediction']}</b>.<br>
                    Model used: <b>{prediction_result['model_name']}</b>.<br>
                    Response time: <b>{elapsed_ms:.0f} ms</b>{confidence_block}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Prediction failed: {prediction_result['error']}")
else:
    st.markdown(
        """
        <div class="result-box">
            Category of your request will appear here after you click
            <b>Make a prediction</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
