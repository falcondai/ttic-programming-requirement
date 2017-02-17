FROM tensorflow/tensorflow:0.12.0-rc1-gpu

MAINTAINER Falcon Dai <me@falcondai.com>

RUN apt-get update && apt-get install -y cmake libopencv-dev python-opencv tmux htop

# dependencies for vizdoom
RUN apt-get install -y build-essential zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev libboost-all-dev

RUN pip install tqdm gym vizdoom

# atari games via OpenAI gym
RUN pip install gym[atari]

WORKDIR /root

# convenient aliases
RUN alias tk="tmux kill-session" && alias tl="tmux list-session"

RUN ["/bin/bash"]
