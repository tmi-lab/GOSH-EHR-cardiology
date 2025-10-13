# Predicting Hospital Stay and Patient Similarity in Paediatric Cardiology using Machine Learning

This repository contains the code, documentation, and resources for the study:

**"Clinically-applicable prediction of hospital stay and patient similarity retrieval in paediatric cardiology using machine learning"**

## 🧠 Overview

This project presents a machine learning framework for:

- Predicting **Length of Stay (LoS)** in paediatric cardiology patients.
- Retrieving **clinically similar patient cases** using EHR data embeddings.
- Supporting **clinical decision-making** through interpretable and deployable models.

The models were trained and validated using data from:
- **Great Ormond Street Hospital (GOSH)** paediatric cardiology admissions (2021–2023)
- **MIMIC-IV** ICU dataset (publicly available)

Key components:
- **Random Forest** model for LoS classification (short ≤3 days vs. long >3 days)
- **BioClinical-BERT** for patient embedding and similarity retrieval
- **Cosine similarity** and **hierarchical ICD/OPCS-based scoring** for patient matching
- **Silent deployment** and **clinical pilot** at GOSH

## 📊 Results

- RF model achieved **91% accuracy** in LoS prediction, outperforming clinicians.
- BioClinical-BERT embeddings enabled meaningful patient similarity retrieval.
- Silent deployment over 6 months showed consistent performance.
- Clinician feedback indicated moderate-to-high utility in real-world settings.

## 🛠️ Installation


git clone https://github.com/<your-org-or-username>/paediatric-cardiology-ml.git
cd paediatric-cardiology-ml
pip install -r requirements.txt
