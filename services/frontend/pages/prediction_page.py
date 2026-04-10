from pathlib import Path
import time
import streamlit as st
from backend_client import DEFAULT_BACKEND_URL, get_models, predict_request
from styles.load_css import STYLES_PATH, load_css

load_css(Path(STYLES_PATH) / "prediction_page.css")

backend_url = st.session_state.get("backend_url", DEFAULT_BACKEND_URL)
models_result = get_models(backend_url)

fallback_models = ["DistilBERT", "GRU", "Logistic Regression"]
model_names = models_result["models"] if models_result["ok"] and models_result["models"] else fallback_models

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

if not models_result["ok"]:
    st.warning(f"Could not load models from backend: {models_result['error']}")

request_text = st.text_area(
    "Request text",
    value="I need to issue a debit card. How can I do this?",
    label_visibility="collapsed",
    key="request_text",
)

left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    selected_model = st.selectbox(
        "Model",
        model_names,
        index=0,
        key="model_select",
        label_visibility="collapsed",
    )
    selected_stage = st.selectbox(
        "Model stage",
        ["Production", "Staging"],
        index=0,
        key="model_stage_select",
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
        start = time.perf_counter()
        prediction_result = predict_request(
            backend_url=backend_url,
            text=clean_text,
            model_name=selected_model,
            model_stage=selected_stage,
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
