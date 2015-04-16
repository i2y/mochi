FROM ubuntu:trusty

MAINTAINER Long Vu "long@tlvu.ca"

RUN locale-gen en_US.UTF-8 && \
    apt-get update && \
    apt-get install -y python-virtualenv python3-dev libzmq3-dev && \
    virtualenv -p /usr/bin/python3 /venv/mochi && \
    . /venv/mochi/bin/activate && \
    LANG=en_US.UTF-8 pip3 install mochi flask Flask-RESTful Pillow && \
    useradd mochiuser --create-home

# prevent encoding errors
ENV LANG=en_US.UTF-8

# best-practice: run as user, not root to avoid security exploit
USER mochiuser
WORKDIR /home/mochiuser

ENTRYPOINT ["/venv/mochi/bin/mochi"]
