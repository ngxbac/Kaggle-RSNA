APP_NAME=ngxbac/pytorch_cv:lastest
CONTAINER_NAME=rsna

run: ## Run container
	nvidia-docker run \
		-e DISPLAY=unix${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
		--ipc=host \
		-itd \
		--name=${CONTAINER_NAME} \
		-v /raid/data/kaggle/rsna/:/data \
		-v /raid/bac/kaggle/logs/:/logs \
		-v $(shell pwd):/kaggle-rsna $(APP_NAME) bash

exec: ## Run a bash in a running container
	nvidia-docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}