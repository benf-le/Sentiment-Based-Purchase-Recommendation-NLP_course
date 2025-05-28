# Sử dụng Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy tất cả file vào Docker image
COPY . /app

# Cài đặt các dependencies cơ bản
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt pip dependencies
RUN pip install --upgrade pip

# Copy riêng requirements.txt nếu có (tối ưu docker caching), nếu không thì cài trực tiếp
RUN pip install streamlit pandas numpy torch torchvision \
    transformers \
    scikit-learn \
    tensorflow keras \
    pyvi \
    pillow \
    streamlit-option-menu

# Nếu bạn có file requirements.txt, thì thay dòng trên bằng:
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Mở port 8501 cho streamlit
EXPOSE 8501

# Lệnh để chạy ứng dụng Streamlit
CMD ["streamlit", "run", "sentimentAnalysis.py", "--server.port=8501", "--server.address=0.0.0.0"]
