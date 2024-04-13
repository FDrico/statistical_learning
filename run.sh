#!/bin/bash

while read line; do export $line; done < .env

docker run --rm \
	-v ./notebooks:/home/jovyan/notebooks -v ./datasets:/home/jovyan/datasets \
	-p 8889:8888 \
	-e JUPYTER_TOKEN=letmein \
	-e CHOWN_EXTRA="/home/jovyan/notebooks,/home/jovyan/datasets" \
    	-e CHOWN_EXTRA_OPTS='-R' \
	-e GRANT_SUDO=yes \
	--user root \
	jupyter \
	start-notebook.py --NotebookApp.token=${ACCESS_TOKEN}


