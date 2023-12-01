# Use the TensorFlow base image
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /home/uoch/holbertonschool_machine_learning/supervised_learning/object_detection

# Copy your package or files into the container
COPY . /home/uoch/holbertonschool_machine_learning/supervised_learning/object_detection

# Install necessary packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev
COPY requirements.txt /home/uoch/holbertonschool_machine_learning/supervised_learning/object_detection
RUN pip install -r /home/uoch/holbertonschool_machine_learning/supervised_learning/object_detection/requirements.txt

# Command to run your package or files
CMD ["python", "on_pc_camera.py"]
