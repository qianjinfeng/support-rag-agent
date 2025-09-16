# Dockerfile
FROM python:3.10-alpine

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --only-binary=all -r requirements.txt && \
    pip cache purge

# 复制代码
COPY . .

# 挂载数据库卷
VOLUME ["/app/chroma_db"]

# 暴露 Streamlit 端口
EXPOSE 8501

# 运行应用
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]