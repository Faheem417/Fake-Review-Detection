# 🕵️‍♂️ Fake Review Detection Using Machine Learning

This project identifies **fake online reviews** using Natural Language Processing (NLP) and machine learning techniques. It helps detect and filter misleading customer reviews automatically.

---

## 📌 Problem Statement

Online platforms are filled with fake or manipulated reviews, which mislead buyers and hurt genuine businesses. This project builds a machine learning model to **detect and classify reviews** as fake or real based on text features.

---

## 🧠 Algorithms Used

- TF-IDF Vectorization
- Logistic Regression
- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)
- XGBoost (optional)

---

## 📊 Model Results

| Metric      | Score   |
|-------------|---------|
| Accuracy    | 94.2%   |
| Precision   | 92.5%   |
| Recall      | 91.0%   |
| F1-Score    | 91.7%   |

✅ These results may vary based on the dataset split and tuning.

---

## 📁 Dataset Info

- Format: CSV file with `Review` and `Label` columns
- Labels: `1` = Fake Review, `0` = Genuine Review
- Source: https://www.kaggle.com/datasets/devmaster347/fake-review-detection-systemamtl

---

## 📂 Project Structure

```bash
Fake-Review-Detection/
├── Fake_Review_Detection.ipynb              # Main Notebook
├── Fake_Review_Detection_Portfolio_Polished_v2.pkl  # Trained Model
├── output.png                               # Model Prediction Screenshot
├── confusion_matrix.png                     # Confusion Matrix Plot
├── accuracy_plot.png                        # Accuracy Visualization
└── README.md                                # Project Description File
🖼️ Screenshots & Visuals
🔍 Confusion Matrix

📈 Accuracy Curve

✅ Final Output Screenshot


🔧 Libraries Used
pandas

numpy

scikit-learn

matplotlib, seaborn

nltk

xgboost (optional)

🚀 Future Improvements
Add deep learning models (BERT, LSTM)

Create web dashboard using Streamlit or Flask

Use real-time review scraping

👤 Author
Abdul Faheem
GitHub: @Faheem417

