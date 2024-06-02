# Rossmann Store Sales Prediction

This repository contains a project for predicting sales of Rossmann stores using machine learning and deep learning techniques.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Model](#machine-learning-model)
- [Deep Learning Model (LSTM)](#deep-learning-model-lstm)
- [Feature Importance & Confidence Interval](#feature-importance--confidence-interval)
- [Model Serialization](#model-serialization)
- [Conclusion](#conclusion)

## Introduction

Predicting sales for Rossmann stores using Random Forest Regressor and LSTM. Steps include data preprocessing, model building, evaluation, and serialization.

## Project Structure

- `data/`: Contains datasets.
- `notebooks/`: Jupyter notebooks for each step.
- `README.md`: Project documentation.

## Data Preprocessing

1. **Loading Data**: Load store, train, and test datasets.
2. **Datetime Conversion**: Convert `Date` columns to datetime format.
3. **Label Encoding**: Encode categorical features.
4. **Feature Engineering**: Extract year, month, day, week, day of the year, and weekend indicator.
5. **Holiday Proximity**: Calculate days to/from holidays.
6. **Scaling**: Standardize features.

## Machine Learning Model

Using Random Forest Regressor:
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
```
Metrics: MSE, MAE, R-squared.

## Deep Learning Model (LSTM)

Steps:
1. **Time Series Preparation**: Transform to supervised learning format.
2. **LSTM Model**: Built with TensorFlow/Keras.
3. **Evaluation**: R-squared value and visual comparison.

## Feature Importance & Confidence Interval

Extract feature importance from Random Forest and estimate confidence intervals using bootstrap sampling.

## Model Serialization

Save models with timestamps for tracking:
```python
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
joblib.dump(pipeline, f'models/random_forest_{timestamp}.pkl')
model.save(f'models/lstm_model_{timestamp}.h5')
```

## Conclusion

This project demonstrates predicting Rossmann store sales using both machine learning and deep learning. Steps include preprocessing, feature engineering, model building, evaluation, and serialization.

