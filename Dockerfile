# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory called as app in the container
WORKDIR /app

# Copy the entire project into app folder into the container
COPY . /app

# Update the package manager and install necessary system tools
RUN apt update -y && apt install awscli -y

# Install application dependencies
RUN pip install -r requirements.txt

# Explicitly install dill (if needed)
RUN pip install dill

# Set the command to run your application
CMD ["python3", "app.py"]