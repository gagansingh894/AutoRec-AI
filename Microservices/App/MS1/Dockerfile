FROM python:3.7-slim
LABEL key="GagandeepSingh"

ENV PYTHONUNBUFFERED 1

#RUN apk add --no-cache bash
#RUN apk add g++
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y wget
RUN pip install Flask dnspython numpy==1.18.0 opencv-python


RUN mkdir /ObjectDetection
WORKDIR /ObjectDetection
COPY ./ObjectDetection /ObjectDetection
RUN wget https://pjreddie.com/media/files/yolov3.weights

RUN useradd -m myapp
USER myapp

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
