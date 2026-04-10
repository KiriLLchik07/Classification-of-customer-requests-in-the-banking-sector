import streamlit as st

page_1 = st.Page("pages/home_page.py", title="Home page")
page_2 = st.Page("pages/prediction_page.py", title="Predict request type")
page_3 = st.Page("pages/models_info_page.py", title="Models info")

pg = st.navigation([ page_1, page_2, page_3])

pg.run()
