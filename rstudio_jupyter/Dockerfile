FROM jupyter:latest
USER root

# Make the user sudoer but without as password, in order to be able to use sudo.
RUN apt install -y sudo 
RUN adduser jovyan sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER jovyan

RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN sudo apt update -y 
RUN sudo apt install -y software-properties-common
RUN sudo add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/"

RUN sudo apt install --no-install-recommends -y r-base
RUN sudo add-apt-repository ppa:c2d4u.team/c2d4u4.0+
RUN sudo apt install --no-install-recommends -y r-cran-rstan
RUN sudo apt install --no-install-recommends -y r-cran-tidyverse

# We need make for R packets later
RUN sudo apt install -y build-essential

# We need to be able to install R packets on the correct lib
RUN sudo usermod -a -G staff jovyan

# Install kernel and other libraries.
RUN sudo Rscript -e "install.packages('IRkernel')"
RUN sudo Rscript -e "install.packages('yhatr')"
RUN sudo Rscript -e "install.packages('ggplot2')"
RUN sudo Rscript -e "install.packages('plyr')"
RUN sudo Rscript -e "install.packages('reshape2')"
RUN sudo Rscript -e "install.packages('forecast')"
RUN sudo Rscript -e "install.packages('stringr')"
RUN sudo Rscript -e "install.packages('lubridate')"
RUN sudo Rscript -e "install.packages('randomForest')"
RUN sudo Rscript -e "install.packages('rpart')"
RUN sudo Rscript -e "install.packages('e1071')"
RUN sudo Rscript -e "install.packages('kknn')"
RUN sudo Rscript -e "install.packages('tidyverse')"
RUN sudo Rscript -e "install.packages('dplyr')"
RUN sudo Rscript -e "install.packages('tidyr')"
RUN sudo Rscript -e "install.packages('timetk')"
RUN sudo Rscript -e "install.packages('readr')"
RUN sudo Rscript -e "install.packages('tidymodels')"
RUN sudo Rscript -e "install.packages('caret')"
RUN sudo Rscript -e "install.packages('ISLR2')"
RUN sudo Rscript -e "install.packages('data.table')"
RUN sudo Rscript -e "install.packages('RCurl')"

# Enable kernel on all users.
RUN R -e "IRkernel::installspec()"

RUN sudo chmod +777 /usr/local/lib/R/site-library
