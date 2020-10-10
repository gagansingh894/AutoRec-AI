FROM python:3.7-slim
LABEL key="GagandeepSingh"

ENV PYTHONUNBUFFERED 1

#RUN apk add --no-cache bash
#RUN apk add g++
RUN pip install Flask dnspython pandas numpy==1.18.0 scikit-learn

RUN mkdir /RecommendationSystem
WORKDIR /RecommendationSystem
COPY ./RecommendationSystem /RecommendationSystem

RUN useradd -m myapp
USER myapp

ENTRYPOINT [ "python" ]
CMD [ "app.py" ]