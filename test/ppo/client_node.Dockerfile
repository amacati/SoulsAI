FROM python:3.9
WORKDIR /home
# Cache requirements install
COPY test/common/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
RUN apt update
RUN apt install swig -y
RUN pip install box2d-py
COPY . /home/SoulsAI
# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
WORKDIR /home/SoulsAI
RUN git checkout dev
RUN python setup.py develop

ENTRYPOINT ["python", "test/ppo/launch_client_node.py"]