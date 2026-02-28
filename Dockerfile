FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN apt-get update && apt-get install -y aria2

RUN mkdir -p /training

WORKDIR /training

COPY . .

