# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and model files
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]

# FROM python:3.10-slim

# ENV PYTHONUNBUFFERED=1 \
#     PORT=7860

# WORKDIR /code

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 7860

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]