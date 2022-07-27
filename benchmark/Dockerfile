FROM python:3.9
WORKDIR /home
# Cache requirements install
ADD SoulsAI /home/SoulsAI
RUN pip install -r /home/SoulsAI/benchmark/requirements.txt
# RUN rm /home/requirements.txt
# COPY /home/amacati/SoulsAI/benchmark/requirements.txt /home/requirements.txt

# Remove all secret files from the container
RUN find /home/SoulsAI -type f -name '*.secret' -delete
RUN git clone https://github.com/amacati/SoulsGym.git
WORKDIR /home/SoulsGym
RUN python setup.py develop
WORKDIR /home/SoulsAI
RUN python setup.py develop
# ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["python", "benchmark/benchmark.py"]
