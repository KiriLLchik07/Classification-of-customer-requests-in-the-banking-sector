import streamlit as st
from pathlib import Path
from services.frontend.styles.load_css import load_css, STYLES_PATH

st.title("Classification of client’s requests in banking sector", text_alignment="center")
st.set_page_config(layout="wide")

load_css(Path(STYLES_PATH) / "home_page.css")

st.markdown(
"""
<br></br>
<div class="about_project">
    This project solve the task of classification client’s requests in banking sector. 
    The system receives the text of the client's request, determines its category, 
    and can be used to automatically route the request to the desired support scenario.
    The following models are used as algorithms for automatically determining the category of goods: Logistic Regression, GRU, DistilBERT.
    The system is implemented by:
        <li> The backend of the service, which is made using the FastAPI library, 
        as well as using the MLflow library for tracking experiments and Mlflow Registry.</li>
        <li> The entire system is run using the Docker Compose network. </li>
</div>
<br></br>
<div class="summary_info_text">
    <h4>
        Summary information about the algorithms used in the system:
    </h4>
</div>
<br></br>
""",
unsafe_allow_html=True)

lr, gru, distil_bert = st.columns(3)

lr.markdown(
"""
<div class="short_info_card">
    <p style="font-weight: bold">Logistic regression</p>
    In conjunction with TF-IDF, the Baseline model is presented. 
    Accuracy of predictions based on the F1-macro metric: 0.866 
</div>
""",
unsafe_allow_html=True)

gru.markdown(
"""
<div class="short_info_card">
    <p style="font-weight: bold">GRU</p>
    The recurrent neural network that handles a text query well as a sequence of tokens. 
    The accuracy of predictions based on the F1-macro metric is 0.875
</div>
""",
unsafe_allow_html=True)

distil_bert.markdown(
"""
<div class="short_info_card">
    <p style="font-weight: bold">DistilBERT</p>
    The model representing the transformer architecture. 
    Demonstrates the best accuracy according to the F1-macro metric: 0.924
</div>
<br></br>
""",
unsafe_allow_html=True)

more_detail_text, more_detail_buttom = st.columns(2)

more_detail_text.markdown(
"""
<div class="page_text">
    More detailed information about the algorithms used in the system can be found on the page
</div>
<br></br>
""",
unsafe_allow_html=True)

more_detail_buttom.button("Models info")

prediction_page_text, prediction_page_buttom = st.columns(2)

prediction_page_text.markdown(
"""
<div class="page_text">
    To determine the type of your request, go to the page
</div>
""",
unsafe_allow_html=True)

prediction_page_buttom.button(
"""
<div class="page_bottom">
    "Prediction"
</div>
""")
