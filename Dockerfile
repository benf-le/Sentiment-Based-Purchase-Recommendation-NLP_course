# Sử dụng Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy tất cả file vào Docker image
COPY . /app


# Cài đặt pip dependencies
RUN pip install --upgrade pip
# Cài đặt các thư viện từ requirements.txt
RUN pip install -r requirements.txt

# Nếu bạn có file requirements.txt, thì thay dòng trên bằng:
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Mở port 8501 cho streamlit
EXPOSE 8501

# Lệnh để chạy ứng dụng Streamlit
CMD ["streamlit", "run", "sentimentAnalysis.py", "--server.port=8501", "--server.address=0.0.0.0"]
