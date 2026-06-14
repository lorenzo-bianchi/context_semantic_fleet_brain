FROM ros:lyrical

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    python3-opencv \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/raysan5/raylib.git /tmp/raylib \
    && mkdir /tmp/raylib/build && cd /tmp/raylib/build \
    && cmake -DBUILD_SHARED_LIBS=ON \
             -DCMAKE_BUILD_TYPE=Release \
             -DPLATFORM=DRM \
             -DOPENGL_VERSION=3.3 .. \
    && make -j$(nproc) && make install && ldconfig \
    && rm -rf /tmp/raylib

RUN pip3 install --no-cache-dir torch torchvision fastapi uvicorn pytest

COPY docker/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
