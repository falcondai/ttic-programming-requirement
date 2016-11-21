FROM tensorflow/tensorflow:0.11.0rc0-gpu

MAINTAINER Falcon Dai <me@falcondai.com>

RUN apt-get update && apt-get install -y cmake

RUN pip install tqdm gym
RUN pip install gym[atari]

WORKDIR /root

RUN ["/bin/bash"]
