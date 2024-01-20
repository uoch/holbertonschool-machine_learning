# Use the locally downloaded Anaconda base image
FROM continuumio/anaconda3:latest

# Set the working directory inside the container
WORKDIR /home/uoch/holbertonschool-machine_learning/unsupervised_learning/hyperparameter_tuning/gpy_notebook

# Update package lists and install necessary system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install packages using Conda and pip
COPY requirements.txt /home/uoch/holbertonschool-machine_learning/unsupervised_learning/hyperparameter_tuning/gpy_notebook
RUN conda install -c conda-forge -y numpy scipy matplotlib
RUN pip install -r /home/uoch/holbertonschool-machine_learning/unsupervised_learning/hyperparameter_tuning/gpy_notebook/requirements.txt || true

# Install Jupyter
RUN conda install -c conda-forge jupyter


# Command to run when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
