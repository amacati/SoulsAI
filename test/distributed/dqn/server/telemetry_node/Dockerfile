FROM python:3.9
WORKDIR /home
# Cache requirements install
COPY test/distributed/dqn/server/telemetry_node/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
COPY . /home/SoulsAI
# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
WORKDIR /home/SoulsAI
RUN git checkout dev
RUN python setup.py develop
ENTRYPOINT ["python", "test/distributed/dqn/server/telemetry_node/launch_node.py"]