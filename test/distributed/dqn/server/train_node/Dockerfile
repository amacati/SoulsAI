FROM python:3.9
WORKDIR /home
# Cache requirements install
COPY test/distributed/dqn/server/train_node/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
COPY . /home/SoulsAI
# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
RUN git clone https://github.com/amacati/SoulsGym.git
WORKDIR /home/SoulsGym
RUN python setup.py develop
WORKDIR /home/SoulsAI
RUN git checkout dev
RUN python setup.py develop
ENTRYPOINT ["python", "test/distributed/dqn/server/train_node/launch_node.py"]