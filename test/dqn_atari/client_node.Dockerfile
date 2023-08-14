FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt update
RUN apt install build-essential -y
WORKDIR /home
# Disable interactive mode for apt
RUN DEBIAN_FRONTEND=noninteractive apt install ffmpeg libsm6 libxext6  -y
RUN apt install swig -y
# Cache requirements install
COPY test/common/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
RUN pip install gymnasium[atari,accept-rom-license,other]
RUN pip install opencv-python
RUN pip install box2d-py
COPY . /home/SoulsAI
# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
WORKDIR /home/SoulsAI
RUN python setup.py develop

ENTRYPOINT ["python", "test/dqn_atari/launch_client_node.py"]