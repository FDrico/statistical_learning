#!/bin/bash

while read line; do export $line; done < .env

docker run --rm  \
	-v /media/fede/Data/repositorios/statistical_learning:/home/jovyan/work \
	-v /media/fede/Data/repositorios/statistical_learning/.jupyter:/home/jovyan/.jupyter \
	-p 8889:8888 \
	jupyter \
	start-notebook.py --NotebookApp.token=${ACCESS_TOKEN}


