# 📰 Fake News Detection System

This is a Machine Learning based web application that detects whether a news article is TRUE or FALSE.

## 📊 Models Used
- Logistic Regression
- Naive Bayes
- Decision Tree
- Passive Aggressive Classifier

The best model is selected automatically based on accuracy.

## 🧠 Dataset Information
- text → News Content
- label → Target
    - 0 = TRUE
    - 1 = FALSE

## ⚙️ How to Run Locally

1. Create virtual environment:
   python -m venv venv

2. Activate:
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Train model:
   cd src
   python train.py

5. Run app:
   streamlit run app.py

## 🚀 Deployment
Deployed using Streamlit Cloud.

---
Author: Riddhi Dwivedi