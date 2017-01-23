FROM tensorflow/tensorflow:0.12.0-rc1-gpu

MAINTAINER Falcon Dai <me@falcondai.com>

RUN apt-get update && apt-get install -y cmake libopencv-dev python-opencv tmux htop

RUN pip install tqdm gym
RUN pip install gym[atari]

WORKDIR /root

RUN ["/bin/bash"]
