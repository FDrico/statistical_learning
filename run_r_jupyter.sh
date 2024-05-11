#!/bin/bash

while read line; do export $line; done < .env

docker run --rm  \
	-v /home/fede/ISL_python:/home/jovyan/work \
	-v /home/fede/ISL_python/.jupyter:/home/jovyan/.jupyter \
	-p 8889:8888 \
	rstudio_jupyter \
	start-notebook.py --NotebookApp.token=${ACCESS_TOKEN}


