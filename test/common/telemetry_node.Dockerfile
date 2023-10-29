FROM python:3.10
WORKDIR /home
# Cache requirements install
COPY soulsai/distributed/server/telemetry_node/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
COPY . /home/SoulsAI/
# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
WORKDIR /home/SoulsAI
RUN python setup.py develop
ENTRYPOINT ["python", "test/common/launch_telemetry_node.py"]