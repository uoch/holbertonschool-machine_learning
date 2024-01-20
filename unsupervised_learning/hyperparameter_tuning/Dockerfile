# Use the TensorFlow base image
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /home/uoch/holbertonschool-machine_learning/unsupervised_learning/hyperparameter_tuning

# Copy your package or files into the container
COPY . /home/uoch/holbertonschool-machine_learning/unsupervised_learning/hyperparameter_tuning

# Install necessary packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev
COPY requirements.txt /home/uoch/holbertonschool-machine_learning/unsupervised_learning/hyperparameter_tuning
RUN pip install -r /home/uoch/holbertonschool-machine_learning/unsupervised_learning/hyperparameter_tuning/requirements.txt

# Command to run your package or files
CMD ["python", "6-bayes_opt.py"]
