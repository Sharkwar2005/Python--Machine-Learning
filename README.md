# FIFA Player Valuation and Predictive Modeling

## ✦ Project Overview
This project focuses on an extensive machine learning pipeline applied to a FIFA player dataset (`Fifa_3.csv`). The primary objective is to execute a rigorous **EDA**, **Preprocessing**, and **transformation** workflow to train **advanced models** capable of accurately predicting player market values and classifications.

## ✦ Problem Statement
Predicting a player's market value requires handling highly variable, right-skewed financial data alongside granular geographical and positional attributes that introduce mathematical noise. The challenge involves structuring a clean **Pipeline** that removes non-predictive identifiers, engineers generalized features, and leverages rigorous **Cross Validation** and **Hyperparameter Tuning** to prevent overfitting across various algorithms.

## ✦ Goals
* **Exploratory Data Analysis (EDA):** Inspect dataset characteristics, evaluate feature distributions, and analyze the extreme right skewness of the `Value Per M$` metric.
* **Preprocessing & Transformation:** Drop the `Name` identifier, filter invalid targets (values $\le 0$), map countries to continents, group field positions into four primary roles, and engineer a 6-tier team quality metric.
* **Hyperparameter Tuning:** Utilize Grid Search and Randomized Search to optimize parameters (e.g., `C`, `gamma`, `max_depth`, `n_neighbors`) across diverse models.
* **Cross Validation & Pipeline Stability:** Evaluate model reliability using K-Fold and Stratified K-Fold techniques embedded within analytical pipelines.

## ✦ Key Insights from the Data Analysis
* **Value Variability:** The dataset highlights extreme right skewness (7.98) in player valuations, led by top outliers such as Kylian Mbappé ($190.5M) and Erling Haaland ($176.5M).
* **Feature Correlations:** `Overall_Rating` (0.56) and `Future Potential` (0.50) are strongly correlated with market value, while age shows a weaker relationship (0.14).
* **Regression Performance:** The **Stacking** Regressor achieved the highest mean cross-validation stability at 0.9760 (std: 0.0076), closely followed by the **Boosting** (Gradient Boosting) Regressor at 0.9748.
* **Classification Performance:** The Gradient **Boosting** Classifier secured the highest mean CV score at 0.9094, with the **Stacking** Classifier achieving a comparable 0.9092.

## ✦ Advanced Models & Methodologies
The project systematically evaluates a wide spectrum of algorithms, progressing from baseline models to complex ensembles:
* **Linear & Distance-Based:** **Logistic Regression** and **KNN** (K-Neighbors Regressor/Classifier).
* **Probabilistic:** **Naïve Bayes variants** (GaussianNB, BernoulliNB, ComplementNB).
* **Margin & Tree-Based:** **Support Vector Machines** (SVC/SVR) and **Decision Tree** (Regressor/Classifier).
* **Ensemble Techniques:** Advanced architectures including the **Random Forest Regressor**, alongside **Voting**, **Boosting** (Gradient Boosting), and **Stacking** ensembles to maximize predictive accuracy.

## ✦ Conclusion
By implementing comprehensive **EDA**, **Preprocessing**, and feature **transformation**, the raw FIFA dataset was successfully optimized for machine learning. Through strict **Cross Validation** and **Hyperparameter Tuning** within a structured **Pipeline**, the project demonstrated that **advanced models**—particularly **Ensemble** methods like **Stacking** and **Boosting**—vastly outperform standalone algorithms like **Logistic Regression**, **Naïve Bayes variants**, **KNN**, **Support Vector Machines**, and baseline **Decision Tree** models. Ultimately, this methodology successfully navigated the heavy skew of player market values to deliver highly stable and accurate predictions.
