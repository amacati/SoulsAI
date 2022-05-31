FROM python:3.9
WORKDIR /home
# Cache requirements install
COPY soulsai/distributed/server/telemetry_node/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
COPY . /home/SoulsAI
WORKDIR /home/SoulsAI
RUN git checkout dev
RUN python setup.py develop
ENTRYPOINT ["python", "soulsai/distributed/server/telemetry_node/launch_node.py"]