FROM docker.io/jupyter/base-notebook

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
