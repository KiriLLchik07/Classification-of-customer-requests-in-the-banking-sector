import streamlit as st
from pathlib import Path

def load_css(path: Path):
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

STYLES_PATH = Path(__file__).resolve().parents[0]
