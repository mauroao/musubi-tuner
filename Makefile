TAG = mauroao/training-musubi:0.1.1

docker-build:
	docker build -t $(TAG) .

docker-push:
	docker push docker.io/$(TAG)

docker-run:
	docker run --rm -it --gpus all --shm-size=8g $(TAG) /bin/bash

