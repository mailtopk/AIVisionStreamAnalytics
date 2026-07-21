FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

CMD ["/bin/bash"]
