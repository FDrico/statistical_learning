Repository based on the book An Introduction to Statistical Learning. A Dockerfile for Jupyternotebook + A Dockerfile for Jupyter+R, along with all notebooks from the text, on Python and R.

# How to use this repository
## For the Jupyter notebook only
On the main directory, run:
```docker build -t jupyter .```
And then run the Docker container with
```./run.sh```

## For the Jupyter+R notebook
On the `rstudio_jupyter` directory, run:
```docker build -t rstudio_jupyter .```
And then run the Docker container from the main directory with
```./run_r_jupyter.sh```
