FROM python:3.12-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1

# ENV VOLC_TTS_TOKEN=xxx
# ENV VOLC_APP_ID=xxx
# ENV VOLC_CLUSTER=volcano_tts
# ENV OPENAI_API_KEY=xxx
# ENV VOICE_TYPE=BV001_streaming
# ENV MAX_LEN=300
# ENV CONCURRENCY=4
# ENV MAX_RETRIES=3

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
