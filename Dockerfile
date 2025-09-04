FROM quay.io/jupyter/base-notebook 

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
USER root
RUN apt update && apt install -y graphviz
USER jovyan
