# app.py
import streamlit as st
from rag_engine import RAGEngine
import json
import os

st.set_page_config(page_title="🔧 Smart Issue Resolver", layout="wide")
st.title("🔧 Smart Issue Resolution System")

@st.cache_resource
def get_rag_engine():
    engine = RAGEngine()
    if not os.path.exists("./chroma_db"):
        with open("data/cases.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        engine.load_data(data)
    return engine

engine = get_rag_engine()

st.sidebar.header("🔧 Device Info (Optional)")
model_type = st.sidebar.text_input("Model Type", value="X300-Pro")
sw_version = st.sidebar.text_input("Software Version", value="V2.3.1")
country = st.sidebar.text_input("Country", value="USA")
components = st.sidebar.multiselect(
    "Affected Components",
    options=["bootloader", "wifi_driver", "touchscreen", "security_module", "power_supply"],
    default=["bootloader"]
)

st.markdown("### 📝 Describe Your Issue")
problem = st.text_area("Please describe the problem", height=150)

if st.button("🔍 Get Solution"):
    if not problem.strip():
        st.warning("Please enter a problem description.")
    else:
        with st.spinner("Searching and generating..."):
            try:
                response = engine.query(
                    question=problem,
                    sw_version=sw_version,
                    model_type=model_type,
                    components=components,
                    country=country
                )
                st.success("✅ Solution generated!")
                st.markdown(response)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")