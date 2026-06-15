FROM my-fleet-brain-base:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    clang-format \
    zsh \
    wget \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

ARG USER_NAME=zero
ARG USER_UID=1000

RUN OLD_USER=$(getent passwd ${USER_UID} | cut -d: -f1) \
    && groupmod -n ${USER_NAME} $(id -gn $OLD_USER) \
    && usermod -l ${USER_NAME} -d /home/${USER_NAME} -m ${OLD_USER} \
    && usermod -s /usr/bin/zsh -aG adm,dialout,plugdev,sudo,tty,video ${USER_NAME} \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV HOME=/home/${USER_NAME}
USER ${USER_NAME}
WORKDIR ${HOME}

RUN mkdir -p ${HOME}/workspace && chown ${USER_NAME}:${USER_NAME} ${HOME}/workspace
RUN mkdir -p ${HOME}/zsh_history && chown ${USER_NAME}:${USER_NAME} ${HOME}/zsh_history

ENV ZSH=${HOME}/.oh-my-zsh
ENV ZSH_CUSTOM=${ZSH}/custom
ENV ZSH_PLUGINS=${ZSH_CUSTOM}/plugins
ENV ZSH_THEMES=${ZSH_CUSTOM}/themes

RUN wget -qO- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | zsh || true

RUN git clone --single-branch --branch 'master' --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_PLUGINS}/zsh-syntax-highlighting \
    && git clone --single-branch --branch 'master' --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_PLUGINS}/zsh-autosuggestions \
    && git clone --single-branch --depth 1 https://github.com/romkatv/powerlevel10k.git ${ZSH_THEMES}/powerlevel10k

COPY --chown=${USER_NAME}:${USER_NAME} ./p10k.zsh ./.p10k.zsh
COPY --chown=${USER_NAME}:${USER_NAME} ./zshrc ./.zshrc

ENV DEBIAN_FRONTEND=dialog

CMD ["zsh"]
