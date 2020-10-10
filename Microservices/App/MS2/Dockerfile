FROM python:3.7-slim
LABEL key="GagandeepSingh"

ENV PYTHONUNBUFFERED 1

#RUN apk add --no-cache bash
#RUN apk add g++
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libgtk2.0-dev
RUN pip install Flask dnspython opencv-python numpy==1.18.0 tensorflow keras requests

RUN mkdir /Classification
WORKDIR /Classification
COPY ./Classification /Classification

RUN useradd -m myapp
USER myapp

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]