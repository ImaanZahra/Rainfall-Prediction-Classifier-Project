# Data Science Project: Building a Rainfall Prediction Classifier

## 🌦️ Project Overview
This project focuses on building and optimizing a machine learning pipeline to predict whether it will rain on a given day in the Melbourne area, Australia. Using historical weather observations (2008–2017) provided by the Australian Bureau of Meteorology, the project explores the entire data science lifecycle: from feature engineering and data cleaning to model optimization using Grid Search Cross-Validation and performance evaluation.

## 🚀 Objectives
* **Feature Engineering:** Transform raw weather data, handle missing values, and create seasonal features from temporal data.
* **Pipeline Construction:** Build a robust `scikit-learn` pipeline that integrates preprocessing (scaling and encoding) with classification models.
* **Model Optimization:** Use `GridSearchCV` to find the best hyperparameters for a Random Forest Classifier.
* **Evaluation:** Interpret model performance using accuracy, precision, recall, F1-score, and confusion matrices.
* **Addressing Data Leakage:** Identify and rename features to ensure the model predicts rainfall based only on information available at the time of prediction.

## 📊 The Dataset
The data is sourced from the [Australian Government's Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) and includes observations from 2008 to 2017. 
* **Original Features:** Min/Max Temperature, Rainfall, Evaporation, Sunshine, Wind Gust Speed/Direction, Humidity, and Cloud cover.
* **Target Variable:** `RainToday` (Binary: Yes/No).
* **Selected Scope:** The analysis specifically targets the Melbourne region by grouping data from **Melbourne**, **Melbourne Airport**, and **Watsonia**.

## 🛠️ Implementation Details

### 1. Data Preprocessing
* **Handling Missing Values:** Rows with missing values in critical columns (like Sunshine and Cloud cover) were dropped to ensure data quality.
* **Target Engineering:** Renamed `RainToday` to `RainYesterday` and `RainTomorrow` to `RainToday` to shift the objective to predicting today's rain based on historical data.
* **Seasonality:** Created a new `Season` feature based on the `Date` column to capture seasonal weather patterns in Australia.

### 2. Machine Learning Pipeline
A structured pipeline was implemented to prevent data leakage during cross-validation:
* **Numerical Transformer:** Standard scaling of continuous variables.
* **Categorical Transformer:** One-Hot encoding for locations, wind directions, and seasons.
* **Classifier:** Random Forest Classifier.

### 3. Hyperparameter Tuning
We optimized the Random Forest model using **Stratified 5-Fold Cross-Validation** over the following grid:
* `n_estimators`: [50, 100]
* `max_depth`: [None, 10, 20]
* `min_samples_split`: [2, 5]

## 📈 Key Results
* **Best Model Accuracy:** ~84% on the unseen test set.
* **Class Imbalance:** The dataset is imbalanced (approx. 76% No rain vs 24% Rain).
* **Feature Importance:** High importance was observed for features like `Humidity3pm`, `Sunshine`, and `Pressure3pm`.
* **Model Sensitivity:** The model achieved a **True Positive Rate (Recall) of ~0.51**, indicating it correctly identifies rain half of the time while maintaining high precision for "No Rain" days.

## 💻 Dependencies
* Python 3.x
* Pandas
* Numpy
* Matplotlib / Seaborn
* Scikit-Learn

## 👥 Authors
* **Jeff Grossman**
* **Abhishek Gagneja**
* **Anita Verma**

---
*© IBM Corporation. All rights reserved.*
