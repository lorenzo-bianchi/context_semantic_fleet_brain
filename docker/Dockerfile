FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-redis \
    libopencv-dev \
    python3-opencv \
    python3-colcon-common-extensions \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-image-transport-plugins \
    ros-jazzy-nav-msgs \
    ros-jazzy-sensor-msgs \
    libasound2-dev \
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxcursor-dev \
    libxinerama-dev \
    libwayland-dev \
    libxkbcommon-dev \
    nlohmann-json3-dev \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/raysan5/raylib.git /tmp/raylib \
    && mkdir /tmp/raylib/build && cd /tmp/raylib/build \
    && cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc) && make install && ldconfig \
    && rm -rf /tmp/raylib

COPY docker/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
