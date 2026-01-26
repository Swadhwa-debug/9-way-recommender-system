# 🔍 9-Way Course Recommender System

**An End-to-End Machine Learning System | Deployed for Global Access**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://9-way-recommender-system.streamlit.app/)
[![Docker Image](https://img.shields.io/badge/Docker-GHCR-blue?logo=docker)](https://github.com/SWMLearner/9-way-recommender-system/pkgs/container/9-way-recommender-system)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 The Project
*Building this system solved a key problem: online learners are overwhelmed by choice but underserved by single-algorithm recommenders. This project explores how **different ML approaches**—from classic clustering to neural networks—perform on the same task, providing a robust, comparative framework for personalized learning.*

## ✨ Live Demos | 全球访问
· **International (Streamlit Cloud):** https://9-way-recommender-system.streamlit.app/
· **China (ModelScope 魔搭社区):** https://modelscope.ai/studios/SargamWadhwa/9-way-recommender-system/

## 📸 Application Interface

| Main Dashboard | Algorithm Tuning |
| :---: | :---: |
| ![App Overview](https://github.com/SWMLearner/9-way-recommender-system/blob/main/assets/app-overview.png) | ![Algorithm Panel](https://github.com/SWMLearner/9-way-recommender-system/blob/main/assets/algorithm-tuning.png) |

*The interactive interface allows users to select recommendation algorithms and adjust parameters in real-time.*

## 🏗️ System Architecture & Production Decisions

This project is built as a **modular and production-ready system**. The core architecture allows for easy comparison and swapping of algorithms, with a focus on stability and user experience in the deployed application.

**Deployed & Active Algorithms (7 Robust Models):**
The following models are fully integrated, tested, and powering the live application:
- **Content-Based:** Course Similarity
- **Clustering:** K-Means, PCA-KMeans
- **Memory-Based:** K-Nearest Neighbors (KNN)
- **Model-Based:** Neural Networks, Regression & Classification with Embeddings

**Temporarily Shelved for Production Stability:**
Two algorithms are disabled in the current deployment, reflecting a **pragmatic engineering choice**:
- **Non-negative Matrix Factorization (NMF):** Awaiting upstream package compatibility with Python 3.13.
- **User Profile Matching:** **Decision:** Removed from the live build due to inconsistent performance and extended debug cycles, prioritizing a reliable user experience.



## 🚀 Deployment & DevOps
The application is **production-containerized** using Docker and deployed via automated CI/CD (GitHub Actions).

**Run the full system locally with one command:**
```bash
docker run -p 8501:8501 ghcr.io/swmlearner/9-way-recommender-system:latest
```
### Build from source:

```bash
git clone https://github.com/SWMLearner/9-way-recommender-system.git
cd 9-way-recommender-system
pip install -r requirements.txt
streamlit run recommender_app.py
```
## 📁 Repository Structure
```bash
├── assets/                    # Application screenshots and visuals
├── recommender_app.py         # Main Streamlit application
├── Recommender.ipynb          # Jupyter notebook with all 9 algorithms
├── utilities.py               # Core data processing & helper functions
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Production container definition
├── .github/workflows/docker.yml  # CI/CD pipeline for automatic builds
└── README.md                  # This documentation
```
## 🔮 Future Enhancements
Ensemble Layer: Implement a meta-recommender that blends results from top-performing base algorithms

Performance Benchmarking: Add A/B testing framework to evaluate recommendation quality in production

Advanced Models: Extend the model hub with Transformer-based recommendation approaches

User Feedback Loop: Incorporate implicit feedback (clicks, time spent) to improve personalization

## 💡 About the Developer
This project bridges my formal training in Statistics (B.Sc. Hons, Delhi University) with my passion for building efficient, practical machine learning systems. It embodies my journey from theoretical math to deployed AI, highlighting skills in MLOps, comparative analysis, and full-stack data science.

Connect with me on GitHub or explore my other projects in market segmentation, fraud detection, and deep learning.

