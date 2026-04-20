# Credit Risk AI

A beginner-friendly Python machine learning project that predicts whether a customer is likely to become delinquent based on basic financial behavior.

## Project Overview

This project demonstrates a complete ML workflow:
- Create and use a synthetic credit-risk dataset
- Train a Logistic Regression model
- Evaluate model performance
- Save the trained model
- Build a Streamlit app for interactive risk prediction

## Features

- Synthetic dataset with 500 realistic customer records
- Binary classification for `delinquent` (0 = no risk event, 1 = likely delinquent)
- Model training script with:
  - Accuracy output
  - Classification report
  - Simple coefficient-based explainability
- Streamlit UI with:
  - Input fields for all model features
  - Predicted risk (High/Low)
  - Delinquency probability
  - Top contributing factors

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Project Structure

```bash
credit-risk-ai/
├── data/
│   └── dataset.csv
├── src/
│   └── train.py
├── app/
│   └── app.py
├── model.pkl              # created after training
├── requirements.txt
└── README.md
```

## How To Run

1. **Clone or download the project**
2. **Create and activate a virtual environment** (recommended)

### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Train the model**

```bash
python src/train.py
```

This creates `model.pkl` in the project root.

5. **Run the Streamlit app**

```bash
streamlit run app/app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

## Business Use Case

Banks, fintech companies, and lending teams can use this workflow as a starting point to:
- Screen applicants faster
- Flag high-risk customers early
- Support credit analysts with data-driven risk indicators
- Build internal tools that combine prediction with simple explainability

> Note: This is a learning project with synthetic data. It is not intended for production lending decisions without stronger data validation, fairness checks, and compliance review.
