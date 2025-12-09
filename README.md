Titanic Survival Prediction

Exploratory Data Analysis (EDA) + Feature Engineering + Machine Learning

This project explores the Titanic dataset using modern data-science techniques.
It includes data cleaning, feature engineering, exploratory visualization, and machine learning models to predict passenger survival.

It is designed as a showcase project for AI/Data Science internships, demonstrating applied skills in Python, EDA, modeling, and evaluation.

📁 Project Structure
"
ML/
│── titanic_eda.ipynb             # Full exploratory analysis & cleaning
│── titanic_modeling.ipynb        # ML modeling pipeline
│── titanic_cleaned.csv           # Final cleaned dataset for modeling
│── README.md                     # Project documentation
"
1. Project Overview

This project analyzes survival patterns on the Titanic using:

Exploratory Data Analysis (EDA)

Feature engineering

Machine learning classification models

Model evaluation metrics

Visualizations (correlation heatmaps, ROC curves, feature importance)

The goal is to understand which factors influenced survival and build predictive models.

2. Data Cleaning & Preprocessing

Steps performed in titanic_eda.ipynb:

✔ Missing Values

age → filled using median

embarked → filled using mode

deck → missing values assigned as Unknown (categorical fix required)

✔ Feature Engineering

New columns created:

Feature	Description
family_size	sibsp + parch + 1
is_alone	1 if family_size==1 else 0
fare_bin	Binned into 4 categories
age_bin	Child / Teen / Adult / Middle-aged / Senior
One-hot encodings	For categorical features (sex, embarked, deck, etc.)
✔ Dropped Columns

Removed redundant columns such as:
class, embark_town, alive, who, alone, and original categorical columns after encoding.

3. Exploratory Data Analysis Highlights

Performed in titanic_eda.ipynb using Seaborn and Matplotlib.

Key findings:

Females had higher survival rate.

First-class passengers had the highest survival.

Being alone reduced the chance of survival.

Higher fare correlated with higher survival.

Younger passengers had slightly better outcomes.

Visualizations include:

Countplots of survival by class, sex, age groups

Correlation heatmap

Boxplots for age and fare

Survival distribution across engineered features

4. Machine Learning Models

Modeling performed in titanic_modeling.ipynb.

✔ Models Implemented
    1) Logistic Regression (Baseline Model)

    Simple, interpretable, strong baseline

    Accuracy: ~81%

    Performs well on linearly separable features

    2) Random Forest Classifier (Ensemble Model)

    Nonlinear, powerful, great with tabular data

    Accuracy: ~78%

    Provides feature importance

5. Model Evaluation
Both models were evaluated using standard machine-learning classification metrics:
accuracy, precision, recall, F1-score, confusion matrix, and AUC (Area Under ROC Curve).

✔ Confusion Matrices & Classification Reports

Logistic Regression

Accuracy: 0.81

AUC Score: 0.848

Excellent at predicting non-survivors (class 0)

Moderate performance on survivors (class 1)

Classification Report:

Metric	        Class 0	            Class 1
Precision      	0.82            	0.80
Recall      	0.89	            0.68
F1-score     	0.85	            0.73


Random Forest

Accuracy: 0.78

AUC Score: 0.828

Slightly lower accuracy but strong at capturing nonlinear relationships

Classification Report:

Metric	        Class 0	   Class 1
Precision   	0.80	   0.74
Recall      	0.85	   0.67
F1-score	    0.83	   0.70


ROC Curve Comparison

The ROC curves for both models were plotted:

Logistic Regression AUC = 0.848

Random Forest AUC = 0.828

Logistic Regression performs slightly better overall but both models show strong discriminatory power.

6. Feature Importance (Random Forest)

Random Forest provides insight into which features influenced survival predictions most.

Top Important Features

sex_male – Being male greatly reduced survival probability

pclass – Higher class passengers survived at higher rates

fare – Higher fare often correlated with higher-class cabins

age – Younger passengers had better chances

family_size – Moderate family sizes showed better outcomes

deck category – Deck location related to survival probability

A horizontal bar plot of the top 15 features is included in titanic_modeling.ipynb.

 6. Feature Importance (Random Forest)

Top predictors of survival include:

sex_male

fare

age

family_size

deck-unknown

pclass

This helps interpret model decision-making.

7. How to Run This Project
Install dependencies:
pip install pandas numpy seaborn matplotlib scikit-learn

Run the notebooks:
jupyter notebook titanic_eda.ipynb
jupyter notebook titanic_modeling.ipynb

8. Future Improvements

Hyperparameter tuning (GridSearchCV)

Add XGBoost or Gradient Boosting models

Use cross-validation for more robust results

Train/test split stratification refinement

Use SMOTE if class imbalance becomes significant

9. Skills Demonstrated

Python (Pandas, Matplotlib, Seaborn, scikit-learn)

Exploratory Data Analysis (EDA)

Data cleaning & preprocessing

Feature engineering

Machine learning (Logistic Regression, Random Forest)

Model evaluation (accuracy, precision, recall, F1, ROC/AUC)

Interpreting model outputs

End-to-end ML workflow creation

Random Forest Feature Importance

Visualization & Insights Extraction