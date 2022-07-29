FROM python:3.9
WORKDIR /home
# Cache requirements install
COPY test/distributed/dqn/client/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt

ENTRYPOINT ["python", "test/distributed/dqn/client/main.py"]