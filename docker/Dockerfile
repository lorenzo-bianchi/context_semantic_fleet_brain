FROM ros:lyrical

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libraylib-dev \
    libopencv-dev \
    python3-opencv \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch torchvision fastapi uvicorn pytest

COPY docker/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
