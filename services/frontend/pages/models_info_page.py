from pathlib import Path
import streamlit as st
from styles.load_css import STYLES_PATH, load_css

load_css(Path(STYLES_PATH) / "models_info_page.css")

st.title("Models info", text_alignment="center")

st.markdown(
    """
    <br></br>
    <div class="info-box">
        On this page, you can learn more about the algorithms used in our system and
        which are used to predict the category of a client's request.
        <br><br>
        Algorithms used in the system:
        <ul>
            <li>Logistic Regression</li>
            <li>GRU</li>
            <li>DistilBERT</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

model_choice = st.radio(
    "Model tabs",
    ["Logistic Regression", "GRU", "DistilBERT"],
    horizontal=True,
    label_visibility="collapsed",
    key="model_info_radio",
)

model_descriptions = {
    "Logistic Regression": (
        "Baseline linear classifier used with TF-IDF features. "
        "Fast, interpretable, and stable for production baselines."
    ),
    "GRU": (
        "Recurrent neural architecture that processes text as token sequences. "
        "Captures sequential context better than simple linear models."
    ),
    "DistilBERT": (
        "Transformer architecture with the highest quality in current experiments. "
        "Best balance between accuracy and inference speed among deep models."
    ),
}

st.markdown(
    f"""
    <div class="model-details-box">
        <p class="model-details-title">{model_choice}</p>
        {model_descriptions[model_choice]}
    </div>
    """,
    unsafe_allow_html=True,
)
