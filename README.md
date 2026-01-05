# 9-Way Recommender System

A machine learning course recommendation system originally implementing 9 different algorithms. Currently deployed with 8 algorithms due to Python 3.13 package compatibility.

## Currently Deployed Algorithms (8):
- Course Similarity
- User Profile Matching  
- Clustering
- Clustering with PCA
- K-Nearest Neighbors (KNN)
- Neural Networks
- Regression with Embedding Features
- Classification with Embedding Features

## Temporarily Disabled:
- Non-negative Matrix Factorization (NMF) - awaiting package compatibility

## Features
- Multiple recommendation algorithms
- Personalized course suggestions  
- Interactive web interface with real-time tuning

## Quick Start
```bash
pip install -r requirements.txt
streamlit run recommender_app.py
```

## 🌍 Live Demos

· International/Streamlit Cloud: https://9-way-recommender-system.streamlit.app/

· China/ModelScope: https://modelscope.ai/studios/SargamWadhwa/9-way-recommender-system/

· Local Run: streamlit run recommender_app.py

## 🐳 Docker Deployment (Production-Ready)

This application is containerized using Docker and automatically built via CI/CD for consistent, reproducible deployment across any environment.

## Run Locally with One Command:

```bash
docker run -p 8501:8501 ghcr.io/swmlearner/9-way-recommender-system:latest
```

Then open http://localhost:8501 in your browser.


## Image Details:

· Registry: GitHub Container Registry (GHCR)

· Image: ghcr.io/swmlearner/9-way-recommender-system:latest

· Build Status: https://github.com/SWMLearner/9-way-recommender-system/actions/workflows/docker.yml/badge.svg

· CI/CD: Automated build on every push to main




