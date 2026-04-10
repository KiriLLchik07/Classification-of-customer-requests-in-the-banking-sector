from pathlib import Path
import streamlit as st
from styles.load_css import STYLES_PATH, load_css

load_css(Path(STYLES_PATH) / "prediction_page.css")

st.title("Prediction section", text_alignment="center")

st.markdown(
    """
    <br></br>
    <div class="info-box">
        Here, the system automatically determines the category of the client's request
        using machine learning algorithms and can redirect it to the appropriate support scenario.
    </div>
    <br></br>
    """,
    unsafe_allow_html=True,
)

request_text = st.text_area(
    "Request text",
    value="I need to issue a debit card. How can I do this?",
    height=80,
    label_visibility="collapsed",
    key="request_text",
)

left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    selected_model = st.selectbox(
        "Model",
        ["DistilBERT", "GRU", "Logistic Regression"],
        index=0,
        key="model_select",
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

predicted_label = "debit_card"
if predict_clicked and request_text.strip():
    st.markdown(
        f"""
        <div class="result-box">
            Category of your request: <b>{predicted_label}</b>.<br>
            Redirecting to the appropriate support department...
        </div>
        """,
        unsafe_allow_html=True,
    )
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
