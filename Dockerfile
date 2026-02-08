FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN python3 -m pip install --upgrade pip

# COPY requirements.txt /app/
COPY requirements.deploy.txt /app/

RUN python3 -m pip install --no-cache-dir -r requirements.deploy.txt
RUN rm -rf /root/.streamlit
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*



COPY video_app.py /app/
RUN mkdir model
COPY model/best.onnx model/
COPY model/nbest.onnx model/



EXPOSE 8502
ENTRYPOINT ["python3", "-m", "streamlit", "run", "video_app.py", "--server.port=8501", "--server.enableCORS=true"]