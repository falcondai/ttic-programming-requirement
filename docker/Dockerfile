FROM tensorflow/tensorflow:1.0.1-gpu

MAINTAINER Falcon Dai <me@falcondai.com>

RUN apt-get update && apt-get install -y cmake libopencv-dev python-opencv tmux htop

# dependencies for vizdoom
RUN apt-get install -y build-essential zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev libboost-all-dev

RUN pip install tqdm gym vizdoom

# atari games via OpenAI gym
RUN pip install gym[atari]

WORKDIR /root

# convenient aliases
RUN echo 'alias tl="tmux list-session"' >> ~/.bashrc
RUN echo 'alias ta="tmux attach-session"' >> ~/.bashrc
RUN echo 'alias tk="tmux kill-session"' >> ~/.bashrc

RUN ["/bin/bash"]
