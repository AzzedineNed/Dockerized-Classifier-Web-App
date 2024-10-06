# Dockerized Classifier Web App

This project is a simple Flask web application that uses a pre-trained ResNet50 model to classify images of cats and dogs. The primary purpose of this project is to explore Docker and containerize a Flask application.

## Project Structure

├── .dockerignore
├── .gitignore 
├── Dockerfile 
├── classifier.py # python script that uses a pre-trained ResNet50 model to classify images of cats and dogs 
├── predict.py # Main Flask app script 
├── requirements.txt # Python dependencies 
├── static # Directory where uploaded images are stored 
├── templates # HTML templates for the web interface
   │ ├── upload.html 
   │ └── result.html


## Features

- **Image Upload:** Users can upload an image through the web interface.
- **Image Classification:** The app uses a ResNet50 model to classify the uploaded image as either a cat or a dog.
- **Dockerized Application:** The entire app is containerized using Docker, making it easy to run in any environment.

## Requirements

- **Docker:** Ensure Docker is installed on your system.
- **Python 3.9** (for local development, if necessary).

## How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/dockerized-classifier-web-app.git
cd dockerized-classifier-web-app


