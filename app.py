# app.py
import streamlit as st
from rag_engine import RAGEngine
import json
import os

st.set_page_config(page_title="🔧 智能问题解决方案推荐系统", layout="wide")
st.title("🔧 智能问题解决方案推荐系统")

# 初始化 RAG 引擎
@st.cache_resource
def get_rag_engine():
    engine = RAGEngine()
    # 如果数据库不存在，导入初始数据
    if not os.path.exists("./chroma_db"):
        with open("data/cases.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        engine.load_data(data)
    return engine

engine = get_rag_engine()

# 用户输入表单
st.sidebar.header("🔧 设备信息（可选）")
model_type = st.sidebar.text_input("机型", value="X300-Pro")
sw_version = st.sidebar.text_input("软件版本", value="V2.3.1")
country = st.sidebar.text_input("国家", value="China")
components = st.sidebar.multiselect(
    "涉及组件",
    options=["bootloader", "wifi_driver", "touchscreen", "security_module", "power_supply"],
    default=["bootloader"]
)

st.markdown("### 📝 描述你的问题")
problem = st.text_area("请输入问题描述，越详细越好", height=150)

if st.button("🔍 获取解决方案建议"):
    if not problem.strip():
        st.warning("请输入问题描述")
    else:
        with st.spinner("正在检索并生成建议..."):
            try:
                response = engine.query(
                    question=problem,
                    sw_version=sw_version,
                    model_type=model_type,
                    components=components,
                    country=country
                )
                st.success("✅ 推荐方案生成完成")
                st.markdown(response)
            except Exception as e:
                st.error(f"❌ 请求失败: {str(e)}")

# 显示数据统计
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 系统状态")
st.sidebar.markdown("- 数据库: Chroma")
st.sidebar.markdown("- 模型: Qwen-Max (API)")
st.sidebar.markdown("- 案例数: 3 (示例)")