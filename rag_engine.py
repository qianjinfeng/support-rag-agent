# rag_engine.py
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
import os

# 初始化嵌入模型（中文推荐 BGE）
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

# 通义千问 API
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")

# Prompt 模板
PROMPT_TEMPLATE = """
你是一个专业的技术支持助手，请根据以下历史案例，为当前问题提供分析和建议。

【当前设备信息】
- 软件版本: {sw_version}
- 机型: {model_type}
- 组件: {components}
- 国家: {country}

【相似历史案例】
{context}

【用户描述的新问题】
"{question}"

请按以下格式回答：

🔍 **问题分析**：
简要分析可能的根本原因。

🛠️ **解决方案建议**：
1. ...
2. ...

📌 **注意事项**：
- 如涉及升级/烧录，请提醒备份。
- 如与国家合规相关，请说明。

🔗 **参考案例**：
{references}

❓ **如信息不足**：
请建议进一步收集哪些信息。
"""

class RAGEngine:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None
        self.llm = Tongyi(model_name="qwen-max", api_key=os.getenv("DASHSCOPE_API_KEY"))
        self.vectorstore = None

    def load_data(self, data):
        """导入历史问题数据"""
        texts = []
        metadatas = []
        ids = []

        for item in data:
            text = f"{item['problem_description']} {item['root_cause']} {item['solution']}"
            texts.append(text)
            metadatas.append(item["metadata"])
            ids.append(item["id"])

        # 创建 Chroma 向量库
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=metadatas,
            ids=ids,
            persist_directory=self.db_path
        )
        print(f"✅ 已导入 {len(data)} 条问题到向量数据库")

    def query(self, question, sw_version=None, model_type=None, components=None, country=None):
        """查询相似问题并生成回答"""
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=embedding_model
            )

        # 构建过滤条件
        where = {}
        if model_type:
            where["model_type"] = model_type
        if sw_version:
            where["sw_version"] = sw_version

        # 检索相似案例（最多3个）
        results = self.vectorstore.similarity_search_with_score(
            question, 
            k=3, 
            where=where
        )

        # 构建上下文
        context = ""
        references = ""
        for doc, score in results:
            context += f"\n【案例 {doc.metadata['id']}】\n问题: {doc.page_content}\n"
            references += f"- {doc.metadata['id']} (相似度: {1-score:.2f})\n"

        # 生成 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("human", PROMPT_TEMPLATE)
        ])
        chain = prompt | self.llm

        response = chain.invoke({
            "question": question,
            "sw_version": sw_version or "未知",
            "model_type": model_type or "未知",
            "components": ", ".join(components) if components else "未知",
            "country": country or "未知",
            "context": context,
            "references": references
        })

        return response