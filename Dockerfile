# 백엔드 배포용 (Hugging Face Spaces Docker SDK — app_port: 7860)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY src src
COPY web web
COPY data/drug_light.db data/drug_light.db
COPY models/ner-drug models/ner-drug
COPY models/intent-drug models/intent-drug
COPY models/kogpt2-drug models/kogpt2-drug

EXPOSE 7860
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "7860"]
