# PhishGuard AI — Phishing Detection System
## Setup & Run Instructions 

---
📌 Overview

PhishGuard AI is a machine learning-based web application designed to detect phishing emails.
It uses Natural Language Processing (NLP) and a trained Naive Bayes model to classify email text as:

✅ Safe Email
⚠️ Phishing Email

The system is built using Flask (Python) for the backend and a simple HTML/CSS interface for the frontend.

---
## 📁 File Structure 

```
phishing_detector/
├── app.py                  ← Flask backend
├── requirements.txt        ← Python packages
├── phishing_model.pkl      ← ✅ Copy from Google Colab
├── vectorizer.pkl          ← ✅ Copy from Google Colab
└── templates/
    └── index.html          ← Frontend UI
```

---

## 🚀 How to Run

#Step 1 — Install packages
pip install -r requirements.txt

#Step 2 — Add model files
Place these inside project root:
   phishing_model.pkl
   vectorizer.pkl

#Step 3 — Run app
python app.py

#Step 4 — Open browser
http://127.0.0.1:5000

## 🔌 API Endpoint

POST /predict

Input:

email_text = "<string>"

Response:

{
  "prediction": "phishing/safe",
  "confidence": 0.98

}

}
>>>>>>> 9c44e81cfafbc895f7c500a3999b6c85a01bde02
