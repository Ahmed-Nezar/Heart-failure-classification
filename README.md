# Heart Failure Classification Project

## Overview
This project applies machine learning techniques to predict the risk of heart failure using the **Heart Failure Prediction Dataset**. It includes an in-depth analysis of the dataset, implementation of multiple machine learning models, and evaluation of their performance using robust metrics. The goal is to assist in the early detection of heart disease and provide insights into patient health.

---

## Objectives

1. **Data Exploration and Visualization**
   - Analyze the dataset to uncover patterns and relationships.
   - Apply Principal Component Analysis (PCA) for dimensionality reduction and visualization.

2. **Data Cleaning and Preprocessing**
   - Handle missing values, outliers, and duplicates.
   - Normalize and scale features as needed.

3. **Model Training and Evaluation**
   - Implement classifiers including Naive Bayes, SVM, KNN, and Decision Trees.
   - Optimize models using Grid Search, Random Search, and Bayesian Optimization (Optuna).

4. **Performance Metrics**
   - Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.

5. **Clustering Analysis**
   - Perform hierarchical clustering and visualize patient groupings using dendrograms.

6. **Documentation and Deliverables**
   - Deliver a comprehensive Jupyter Notebook and a detailed report.

---

## Dataset Description

### Source
The dataset consists of clinical and demographic attributes of patients, such as:

- **Age**
- **Sex**
- **Chest Pain Type**
- **Resting Blood Pressure (RestingBP)**
- **Cholesterol**
- **Fasting Blood Sugar (FastingBS)**
- **Resting ECG Results**
- **Maximum Heart Rate (MaxHR)**
- **Exercise-Induced Angina (ExerciseAngina)**
- **ST Depression (Oldpeak)**
- **ST Segment Slope (ST_Slope)**
- **Heart Disease (Target)**

---

## Exploratory Data Analysis (EDA)

- **Duplicate and Missing Values**: Confirmed none.
- **Visualizations**: 
  - KDE plots for numerical features.
  - Pie charts for categorical distributions.
  - Scatter plots for feature relationships.
  - Correlation matrix for numerical features.

---

## Preprocessing Scenarios

Multiple scenarios were designed to test preprocessing techniques, including:

1. **Baseline Dataset**
2. **Feature Selection**
3. **Handling Missing Values**
4. **Outlier Removal**
5. **Feature Engineering**
6. **Duplicate Removal**
7. **Data Splitting**
8. **Advanced Feature Engineering**

Each scenario had normalized (N) and scaled (S) variants.

---

## Machine Learning Models

1. **Naive Bayes**
   - Probabilistic model suitable for categorical and numerical data.
2. **Support Vector Machines (SVM)**
   - Effective for high-dimensional datasets with kernel-based flexibility.
3. **K-Nearest Neighbors (KNN)**
   - Instance-based algorithm relying on distance metrics.
4. **Decision Trees**
   - Interpretable and flexible model using hierarchical data splits.

---

## Performance Evaluation

Metrics used to evaluate models:

- **Accuracy**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **Class-wise Metrics**

---

## Optimization Techniques

1. **Grid Search**: Exhaustive search over parameter space.
2. **Random Search**: Random sampling for efficiency.
3. **Optuna (Bayesian Optimization)**: Probabilistic guidance for optimal parameter search.

---

## Results and Insights

- **Best Models**:
  - Naive Bayes: Fast and robust with minimal preprocessing.
  - SVM: High accuracy with standardized preprocessing.
  - KNN: Effective for local data patterns after tuning.
  - Decision Trees: Competitive accuracy with interpretable results.

- **Key Findings**:
  - Preprocessing has a significant impact on performance.
  - Optuna consistently outperformed other tuning methods.

---

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Scikit-learn, Optuna
- **Environment**: Jupyter Notebook

---

## Deliverables

1. **Code Implementation**: A well-documented Jupyter Notebook.
2. **Report**: Detailed documentation of methods, results, and insights.
3. **Saved Models**: Optimized models stored as `.pkl` files for reproducibility.

---

## Authors

- [Ahmed Nezar Ahmed](https://github.com/Ahmed-Nezar)  
- [AbdulRahman Hesham Kamel](https://github.com/AHKSASE2002)  
- [Kirollos Ehab Magdy](https://github.com/KirollosEMH)

---

## Acknowledgments

This project was completed as part of the Machine Learning course (CSE381) at Ain Shams University, Faculty of Engineering.

