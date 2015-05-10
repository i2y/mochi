FROM ubuntu:trusty

MAINTAINER Long Vu "long@tlvu.ca"

RUN locale-gen en_US.UTF-8 && \
    apt-get update && \
    apt-get install -y curl git make g++ libbz2-dev libreadline-dev libssl-dev libsqlite3-dev python3-dev libzmq3-dev && \
    useradd mochiuser --create-home

# best-practice: run as user, not root to avoid security exploit
USER mochiuser

RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer > $HOME/pyenv-installer && \
    bash $HOME/pyenv-installer && \
    export PATH="$HOME/.pyenv/bin:$PATH" && \
    pyenv init - && \
    pyenv virtualenv-init - && \
    pyenv install 3.4.3 && \
    pyenv rehash

# other python versions, insert another RUN here to re-use existing cache

RUN export PATH="$HOME/.pyenv/bin:$PATH" && \
    eval "$(pyenv init -)" && \
    eval "$(pyenv virtualenv-init -)" && \
    pyenv global 3.4.3 && \
    pyenv virtualenv venv343mochi && \
    pyenv activate venv343mochi && \
    pip3 install --upgrade pip && \
    LANG=en_US.UTF-8 pip3 install mochi flask Flask-RESTful Pillow && \
    mkdir $HOME/workdir

# prevent encoding errors
ENV LANG=en_US.UTF-8

WORKDIR /home/mochiuser/workdir

ENTRYPOINT ["/home/mochiuser/.pyenv/versions/venv343mochi/bin/mochi"]
