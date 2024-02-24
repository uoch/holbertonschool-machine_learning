# Use TensorFlow GPU base image
FROM tensorflow/tensorflow:2.13.0-gpu

# Install additional dependencies
RUN apt-get update && \
    apt-get install -y swig cmake && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code into the container
COPY . /app

# Command to run your application
CMD ["python", "reinforcement_learning/deep_q_learning/train.py"]
