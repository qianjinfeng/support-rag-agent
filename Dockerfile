# Dockerfile
FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip cache purge && pip install --only-binary=all -r requirements.txt 

COPY . .

VOLUME ["/app/chroma_db"]

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]