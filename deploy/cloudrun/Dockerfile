FROM python:3.12-slim
# OpenCascade's viewer libraries load at import even without a display; libgomp for torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio==6.31.0
COPY . .
ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_ANALYTICS_ENABLED=False MPLBACKEND=Agg
# Cloud Run passes the port to listen on in $PORT.
CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-8080} exec python app.py"]
