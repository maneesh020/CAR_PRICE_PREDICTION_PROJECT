# Car Price Prediction Using Machine Learning

## Introduction
This project is designed to predict car prices in the American market using machine learning models. The main goal is to assist a Chinese automobile company in understanding the key factors that influence car prices, enabling them to make better decisions when launching their products in the U.S. market.

## Problem Statement
The company wants to:
- Identify the key factors that influence car prices.
- Build predictive models to estimate car prices based on car features.
- Gain insights into the U.S. automobile market to modify their business strategy accordingly.

## Data Overview
The dataset contains multiple features that influence car prices, including:
- **Car Model**
- **Engine Size**
- **Fuel Type**
- **Horsepower**
- **Car Brand**
- **Mileage**
- **Price (Target Variable)**

## Project Goals
The primary goals of this project are:
- To develop machine learning models for predicting car prices.
- To compare and evaluate the performance of different models.
- To perform feature importance analysis to understand which variables most influence car prices.
- To fine-tune the best model using hyperparameter tuning for better accuracy.

## Tools and Libraries Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **IDE:** Jupyter Notebook

## Methodology
The following steps were followed to complete the project:

### Step 1: Data Collection
- The dataset was provided through a market survey.
- It was imported using Pandas.

### Step 2: Data Preprocessing
- Handled missing values by imputation.
- Converted categorical variables to numerical using One-Hot Encoding.
- Split the data into training and testing sets.

### Step 3: Model Building
Implemented the following regression models:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor (SVR)

### Step 4: Model Evaluation
- Evaluated the models using performance metrics like:
  - R2 Score
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
- Compared the performance of all models using bar charts.

### Step 5: Feature Importance
- Used Random Forest Regressor to determine the most important features impacting car prices.
- Visualized the feature importance using bar charts.

### Step 6: Hyperparameter Tuning
- Applied GridSearchCV on the best-performing model to optimize its hyperparameters.
- Improved model performance after tuning.

## Results and Insights
- The Random Forest Regressor was found to be the most accurate model.
- The most significant factors affecting car prices were engine size, horsepower, and car brand.
- After hyperparameter tuning, the model accuracy increased significantly.

## Folder Structure
```plaintext
Car Price Prediction
│
├── Car_Price_Prediction_Implementation.ipynb   # Jupyter Notebook
├── README.md                    # Project Documentation
├── CarPrice_Dataset.csv                     # Dataset File
```

## Key Recommendations
- Focus on cars with optimized engine size and horsepower to achieve competitive pricing.
- Develop energy-efficient and high-performance vehicles for higher market penetration.
- Collect more market data to improve the model's performance.

## Future Scope
- Expand the model by integrating deep learning algorithms.
- Include external economic factors like fuel prices, demand trends, and inflation.
- Deploy a web-based application for real-time car price predictions.

## Author
- **Name:** Maneeshkumar G 

This project provided actionable insights into car pricing in the U.S. market and successfully built a predictive model to assist the company in making informed business decisions.
