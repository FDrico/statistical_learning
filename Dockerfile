FROM quay.io/jupyter/base-notebook 
USER root
RUN groupadd -g 1000 hostgroup && \
    usermod -a -G hostgroup jovyan
RUN chown -R jovyan:hostgroup /home/jovyan

USER jovyan
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
