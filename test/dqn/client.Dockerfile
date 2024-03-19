FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update
RUN apt-get install build-essential -y
WORKDIR /home
# Cache requirements install
COPY test/dqn/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
RUN apt update
RUN apt install swig -y
RUN pip install box2d-py
COPY . /home/SoulsAI
# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
WORKDIR /home/SoulsAI
RUN pip install -e .

ENTRYPOINT ["python", "test/dqn/launch_client.py"]