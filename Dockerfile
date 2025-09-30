# Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y git wget curl build-essential rsync && apt-get clean

WORKDIR /app

# Install flow-cutter for TDP
RUN git clone https://github.com/ben-strasser/flow-cutter-pace20.git && \
    cd flow-cutter-pace20 && \
    ./build.sh
ENV PATH="/app/flow-cutter-pace20:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY . /app

# CMD ["python", "main.py"]
