FROM nvidia/cuda:11.8.0-base-ubuntu22.04
WORKDIR /home
# Cache requirements install
COPY soulsai/distributed/server/training_node/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
COPY . /home/SoulsAI/
# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
WORKDIR /home/SoulsAI
RUN python setup.py develop
ENTRYPOINT ["python", "test/dqn_atari/launch_training_node.py"]