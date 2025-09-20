# Wine Classification MLOps

This project implements a **Wine Classification Model** using a **Random Forest Classifier** to predict the class of wine based on various chemical properties. The model is exposed via an API using **FastAPI** and interacts with a **Streamlit** web application for user input and predictions. The entire project is containerized using **Docker** and orchestrated with **Docker Compose**.

## Model Overview

The model classifies wines into three categories based on their chemical attributes such as **alcohol content**, **malic acid**, **ash**, **magnesium**, and other chemical features. The dataset used for training is the **Wine dataset** from **UCI** repository, which consists of 13 features for each wine sample.

- **Model Type**: Random Forest Classifier
- **Classes**: 3 (corresponding to the type of wine)
- **Dataset**: UCI Wine dataset

## Setup and Run

### Prerequisites

Ensure you have the following installed:
- Docker
- Docker Compose

### Steps to Run Locally

1. **Clone the repository**:

   ```bash
   git clone https://github.com/leryamerlennn/wine-classification-mlops.git
   cd wine-classification-mlops/code/deployment
2. **Build and Run the Docker containers:**

   ```bash
   docker compose up --build

**Access the Web Application**
- Streamlit web app will be available at http://localhost:8501
- FastAPI will be available at http://localhost:8000
