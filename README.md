
# Business Analytics Project – E-Commerce Customer Churn Prediction Using Machine Learning

This project analyzes customer behavior data to predict **churn** (whether a customer is likely to leave) for an e-commerce platform.  
It was developed as part of a Business Analytics module to help businesses identify at-risk customers and take proactive retention actions.

## 🔍 Problem Statement

Customer churn is a major challenge for e-commerce companies. The goal is to build a machine learning model that can accurately predict if a customer will churn based on behavior, activity, preferences, and spending data.

## 📁 Dataset

- **File**: `ecommerce_churn.csv`
- **Description**: Contains customer demographics, order behavior, complaint status, and payment preferences.
- **Features used**:
  - Age, Gender, PreferredLoginDevice, PreferredPaymentMode, PreferedOrderCat
  - Time on app, Number of Purchases, Last Purchase Days Ago
  - Total Spend, Complain status, and more
- **Target variable**: `Churn` (0 = No churn, 1 = Churn)

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn (Visualizations)
- Scikit-learn (Modeling, preprocessing, evaluation)
- Imbalanced-learn (SMOTE for imbalance handling)

## 🧠 Model Used & Performance

- **Model**: Random Forest Classifier (tuned using GridSearchCV)
- **Accuracy**: ~92% on test data
- **ROC-AUC Score**: 0.95
- **Log Loss**: 0.18

The final model was tuned using hyperparameter optimization and evaluated using confusion matrix, ROC-AUC, and log loss.

## 📊 Visualizations

- Customer behavior by preferred device, category, payment mode
- Top spenders
- Feature distributions (time on app, complaints, age)
- Confusion matrix of final model

## 📂 Files

```
📦 ecommerce-churn-prediction
├── ecommerce_churn.csv
├── ecommerce_churn.py
├── README.md
```

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/chitra-cp/ecommerce-churn-prediction.git
```

2. Run the script:
```bash
python ecommerce_churn.py
```

3. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## 👩‍💻 Author

**Chitra S**  
_M.Sc. Data Analytics Graduate_
