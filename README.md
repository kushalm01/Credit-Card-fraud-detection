# Credit Card Fraud Detection 

Welcome! This project focuses on **detecting fraudulent credit card transactions** using machine learning. Detecting fraud is challenging because fraudulent transactions are extremely rare compared to legitimate ones, making this a classic **imbalanced classification problem**.

The dataset used is the **[Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)** from Kaggle. It contains anonymized transaction features and a target label (`Class`), where:

- `0` = Legitimate transaction  
- `1` = Fraudulent transaction  

## What’s Inside

### Data Exploration & Visualization
- Overview of dataset using `.info()` and `.describe()`.  
- Check class imbalance (`fraud` vs `non-fraud`).  
- Correlation heatmaps and scatter plots to explore feature relationships.  

### Feature Engineering
- Time-based features: `hour`, `day`, `month`, `dayofweek`.  
- Drop original datetime after conversion.  

### Machine Learning Models
We train and compare multiple models to detect fraud:

- **Decision Tree**  
- **Random Forest**  
- **XGBoost**  
- **Logistic Regression**

Each model is evaluated using **recall**, which is key in fraud detection since missing a fraud is costlier than a false alarm.

### Handling Class Imbalance
Techniques to improve model performance on rare fraud cases:

- **Random Undersampling**  
- **SMOTE (Synthetic Minority Oversampling Technique)**  

### Feature Importance
- Visualize the **top features** driving predictions for each model.  
- Helps interpret model decisions and understand patterns in fraudulent transactions.  

## How to Run

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn kagglehub
```

## Preprocess & Train

- **Feature engineering**: create time-based features (`hour`, `day`, `month`, `dayofweek`).  
- **Train models**: Decision Tree, Random Forest, XGBoost, Logistic Regression.  
- **Handle class imbalance**: use SMOTE or random undersampling.  
- **Scale features**: apply `StandardScaler`.  
- **Evaluate models**: focus on **recall score**.  
- **Visualize feature importance**: plot top features driving model predictions.  

## Key Takeaways

- Handling **class imbalance** is crucial for detecting fraud effectively.  
- **XGBoost** and **Random Forest** generally perform well; Logistic Regression improves significantly with SMOTE.  
- Time-based features can reveal hidden fraud patterns in transactions.  

## Future Work

- Explore **deep learning approaches** such as neural networks or autoencoders for anomaly detection.  
- Experiment with **ensemble methods** combining multiple models.  
- Consider **real-time deployment** of fraud detection systems.

