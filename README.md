# MSc_Data_Science_Project

# Bank Marketing Subscription Prediction

## Project Overview

This project applies supervised machine learning techniques to predict whether a customer will subscribe to a term deposit based on data from a banking marketing campaign. The objective is to compare multiple classification models and evaluate their effectiveness, particularly in the presence of class imbalance.

The project follows a complete machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), model development, imbalance handling, and performance evaluation.

---

## Dataset

The dataset used is the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

- Records: 45,211  
- Features: 17 (before preprocessing)  
- Type: Tabular data (numerical + categorical)  
- Target Variable: `y` (subscription: yes/no)

Dataset link:  
https://archive.ics.uci.edu/ml/datasets/bank+marketing

---

## Project Structure

├── bank-full.csv
├── DS_Project_Srujan.ipynb
├── MSc Data Science Project Report
├── README.md
├── bank.zip


---

## Methodology

The project follows these key steps:

### 1. Data Preprocessing
- Handling missing values (`unknown`)
- One-hot encoding for categorical variables
- Feature scaling using StandardScaler
- Train-test split (80:20, stratified)

### 2. Exploratory Data Analysis (EDA)
- Distribution of numerical and categorical features
- Target variable imbalance analysis
- Feature-target relationships

### 3. Models Implemented
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Random Forest (Balanced - class weighting)
- Random Forest (Undersampled)

### 4. Handling Class Imbalance
- Class weighting (cost-sensitive learning)
- Random undersampling (data-level approach)

---

## Evaluation Metrics

Due to class imbalance, multiple metrics were used:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Precision–Recall Curve  

---

## Results Summary

- Random Forest achieved the best overall performance  
- Undersampling improved recall but reduced precision  
- Class weighting had limited impact compared to standard Random Forest  
- Ensemble methods performed better than individual models  

---

## Key Insights

- Class imbalance significantly affects model performance  
- Recall is critical for detecting subscribers  
- Trade-offs exist between precision and recall  
- Proper preprocessing and evaluation are essential  

---

## Requirements

To run the notebook, install the following:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
