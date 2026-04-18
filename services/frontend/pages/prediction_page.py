from pathlib import Path
import time
import streamlit as st
from backend_client import (
    DEFAULT_BACKEND_URL,
    get_mlflow_models,
    get_model_info,
    get_models,
    predict_request,
)
from styles.load_css import STYLES_PATH, load_css

load_css(Path(STYLES_PATH) / "prediction_page.css")

backend_url = st.session_state.get("backend_url", DEFAULT_BACKEND_URL)
loaded_models = get_models(backend_url)
mlflow_models = get_mlflow_models(backend_url)

loaded_model_names = loaded_models["models"] if loaded_models["ok"] else []
registry_model_names = mlflow_models["model_names"] if mlflow_models["ok"] else []

all_model_options = list(dict.fromkeys(loaded_model_names + registry_model_names))
if not all_model_options:
    all_model_options = ["Banking77_DistilBERT"]

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
    st.warning(f"Could not load loaded-model list from backend: {loaded_models['error']}")
if not mlflow_models["ok"]:
    st.warning(f"Could not load model list from MLflow Registry: {mlflow_models['error']}")

request_text = st.text_area(
    "Request text",
    label_visibility="collapsed",
    key="request_text",
)

left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    selected_model = st.selectbox(
        "Model",
        all_model_options,
        index=0,
        key="model_select",
        label_visibility="collapsed",
    )
    selected_alias = st.selectbox(
        "Model alias",
        ["production", "reserve", "baseline"],
        index=0,
        key="model_alias_select",
        label_visibility="collapsed",
    )

with right_col:
    st.markdown(
        f"""
        <div class="hint-box">
            <span class="hint-model">{selected_model}</span> is currently selected for prediction.
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
        model_info = get_model_info(
            backend_url=backend_url,
            model_name=selected_model,
            alias=selected_alias,
        )
        if not model_info["ok"]:
            st.warning(
                f"Alias '{selected_alias}' is not available for model '{selected_model}'. "
                f"Details: {model_info['error']}"
            )
        else:
            start = time.perf_counter()
            prediction_result = predict_request(
                backend_url=backend_url,
                text=clean_text,
                model_name=selected_model,
                model_alias=selected_alias,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            if prediction_result["ok"]:
                confidence = prediction_result.get("confidence")
                prediction_label = prediction_result.get("prediction_label") or prediction_result.get("prediction")
                prediction_code = prediction_result.get("prediction_code")
                confidence_block = f"<br>Confidence: <b>{confidence:.3f}</b>" if confidence is not None else ""
                code_block = (
                    f'Request code: <b>{prediction_code}</b>.'
                    if prediction_code is not None
                    else "Request code: <b>n/a</b>."
                )
                st.markdown(
                    f"""
                    <div class="result-box">
                        Category of your request: "<b>{prediction_label}</b>". {code_block}<br>
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
