FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /home
# Cache requirements install
COPY soulsai/distributed/server/training_node/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
COPY . /home/SoulsAI
# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
RUN git clone https://github.com/amacati/SoulsGym.git
WORKDIR /home/SoulsGym
RUN python setup.py develop
WORKDIR /home/SoulsAI
RUN python setup.py develop
ENTRYPOINT ["python", "soulsai/distributed/server/training_node/launch_node.py"]