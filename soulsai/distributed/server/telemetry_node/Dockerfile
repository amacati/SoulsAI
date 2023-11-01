FROM python:3.10
WORKDIR /home
# Cache requirements install
COPY soulsai/distributed/server/telemetry_node/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
ENV MPLCONFIGDIR=/var/cache/matplotlib
COPY . /home/SoulsAI
# Remove git files from the container since they are not needed and WandB will complain
RUN rm -rf /home/SoulsAI/.git
# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
WORKDIR /home/SoulsAI
RUN python setup.py develop
ENTRYPOINT ["python", "soulsai/distributed/server/telemetry_node/launch.py"]