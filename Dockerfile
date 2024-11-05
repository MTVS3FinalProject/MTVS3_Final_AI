# 베이스 이미지 설정 (CUDA 12.5 지원 Ubuntu)
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# 필요한 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nginx \
    && ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/BASE_DIR && chmod -R 777 /app/BASE_DIR

# PyTorch 설치 (CUDA 지원 버전)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# FastAPI 및 Uvicorn 설치
RUN pip3 install uvicorn fastapi

# Timm 라이브러리 호환 버전 설치
RUN pip3 install timm==0.6.12

# tf-keras 설치
RUN pip3 install tf-keras

# 작업 디렉터리 설정
WORKDIR /app

# face.txt 복사 및 필요한 패키지 설치
COPY face.txt .
RUN pip3 install --no-cache-dir -r face.txt

# main.py 애플리케이션 파일 복사
COPY main.py /app

# Nginx 설정 파일 복사
COPY nginx.conf /etc/nginx/nginx.conf

# ComfyUI 클론 및 의존성 설치
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    pip install --upgrade pip && \
    pip install -r /app/ComfyUI/requirements.txt && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git /app/ComfyUI/custom_nodes/ComfyUI-Manager

# CUDA 환경 변수 설정
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
