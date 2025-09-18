# rag_engine.py
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import os

# 更新后的 rag_engine.py 核心部分


# ✅ 使用新方式创建 embedding
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"
)

# ✅ 使用 OllamaLLM（新类）
llm = OllamaLLM(
    model="llama3.2:1b",
    base_url="http://host.docker.internal:11434",
    temperature=0.2
)

# 后续逻辑不变...

PROMPT_TEMPLATE = """
You are a professional technical support assistant. Please analyze the user's issue and provide recommendations.

【Device Context】
- Software Version: {sw_version}
- Model Type: {model_type}
- Components: {components}
- Country: {country}

【Relevant Historical Cases】
{context}

【User's Problem】
"{question}"

Please respond in the following format:

🔍 **Problem Analysis**:
Briefly explain the possible root cause.

🛠️ **Recommended Solution**:
1. Step-by-step actions.
2. If multiple, list clearly.

📌 **Notes**:
- Remind to back up before updates.
- Mention compliance if needed.

🔗 **Reference Cases**:
{references}

❓ **If More Info Needed**:
Suggest what logs or details to collect.
"""

class RAGEngine:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
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

        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embedding_model,
            metadatas=metadatas,
            ids=ids,
            persist_directory=self.db_path
        )
        print(f"✅ Loaded {len(data)} cases into vector DB")

    def query(self, question, sw_version=None, model_type=None, components=None, country=None):
        """查询并生成回答"""
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=embedding_model
            )

        # 构建 filter 使用 ChromaDB 正确的操作符语法
        filters = []
        if model_type:
            filters.append({"model_type": {"$eq": model_type}})
        if sw_version:
            filters.append({"sw_version": {"$eq": sw_version}})
        # 可选：添加 components 或 country 过滤
        # if country:
        #     filters.append({"country": {"$eq": country}})

        # 只有当有至少一个条件时，才使用 $and
        if len(filters) == 0:
            # 无过滤条件：不传 filter
            kwargs = {"k": 3}
        elif len(filters) == 1:
            # 单个条件：直接传（不需要 $and）
            kwargs = {"k": 3, "filter": filters[0]}
        else:
            # 多个条件：用 $and 包裹
            kwargs = {"k": 3, "filter": {"$and": filters}}

        results = self.vectorstore.similarity_search_with_score(
            question,
            **kwargs
        )

        context = ""
        references = ""
        for doc, score in results:
            context += f"\n[Case {doc.metadata['id']}]\n{doc.page_content}\n"
            references += f"- {doc.metadata['id']} (similarity: {1-score:.2f})\n"

        prompt = ChatPromptTemplate.from_messages([("human", PROMPT_TEMPLATE)])
        chain = prompt | llm

        response = chain.invoke({
            "question": question,
            "sw_version": sw_version or "Unknown",
            "model_type": model_type or "Unknown",
            "components": ", ".join(components) if components else "Unknown",
            "country": country or "Unknown",
            "context": context,
            "references": references
        })

        return response