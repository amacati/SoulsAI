FROM python:3.9
WORKDIR /home
# Cache requirements install
COPY soulsai/distributed/server/requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt
RUN rm /home/requirements.txt
COPY . /home/SoulsAI
RUN git clone https://github.com/amacati/SoulsGym.git
WORKDIR /home/SoulsGym
RUN git checkout dev
RUN python setup.py develop
WORKDIR /home/SoulsAI
RUN git checkout dev
RUN python setup.py develop
ENTRYPOINT ["python", "soulsai/distributed/server/main.py"]