# Bank Customer Churn Prediction Model

**Role:** Data Science & Machine Learning Project  
**Author:** Vinay Kumar Chinthamalla  
**Notebook:** [`Bank_Customer_Churn_Model.ipynb`](./Bank_Customer_Churn_Model.ipynb)

---

## Project Overview
Customer attrition (churn) is one of the most critical revenue challenges in retail banking. This project designs, optimizes, and validates a supervised machine learning pipeline to identify banking customers at high risk of churning, enabling proactive retention strategies. Tested on a dataset of **10,000 customer records and 13 behavioral/financial features**, the optimized **Support Vector Machine (SVM)** classifier achieves **92% prediction accuracy**.

---

## Key Tasks & Technical Workflow

1. **Exploratory Data Analysis & Preprocessing (`10,000` Rows, `13` Features):**
   - Analyzed customer demographics, credit scores, account balances, product holdings, tenure, and activity status.
   - Performed categorical encoding and numerical feature standardization (`StandardScaler`) for smooth integration into distance- and margin-based classifiers.

2. **Class Imbalance Mitigation:**
   - Addressed severe target class imbalance between retained and churned customers using **Random Under-Sampling (RUS)** and **Random Over-Sampling (ROS)** to prevent majority-class bias and improve recall on churners.

3. **Model Optimization & Validation (`92%` Accuracy):**
   - Trained and compared baseline and resampled classifiers (`Support Vector Machine (SVM)`).
   - Applied **Grid Search (`GridSearchCV`)** cross-validation to tune hyperparameters (`C`, `gamma`, and kernel functions), achieving **92% prediction accuracy** and balanced precision/recall across classes.

---

## Technologies Used
- **Language:** `Python`
- **Data Manipulation:** `Pandas`, `NumPy`
- **Machine Learning:** `Scikit-learn` (`SVC`, `GridSearchCV`, `StandardScaler`, resampling utilities)
- **Data Visualization:** `Matplotlib`, `Seaborn`
