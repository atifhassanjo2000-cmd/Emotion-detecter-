FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf-cache
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from transformers import AutoImageProcessor, AutoModelForImageClassification as M; n='dima806/facial_emotions_image_detection'; AutoImageProcessor.from_pretrained(n); M.from_pretrained(n)"
EXPOSE 10000
CMD ["python", "app.py"]
