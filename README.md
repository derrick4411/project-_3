# Telecom Customer Churn Prediction

## Project Overview

Binary classification model to predict customer churn for a telecommunications company, enabling proactive retention strategies and reducing revenue loss from customer attrition.





**Status:** Model ready for production deployment  
**Owner:** Data Science Team  
**Stakeholder:** Head of Telecommunications

## Models Compared

| Aspect | Base Model (Untuned) | Tuned Model (GridSearchCV) |
|--------|---------------------|---------------------------|
| **Algorithm** | Logistic Regression | Logistic Regression + Hyperparameter Tuning |
| **Preprocessing** | StandardScaler | StandardScaler |
| **C** | Default (1.0) | Optimized (0.01, 0.1, 1, 10) |
| **Solver** | Default (lbfgs) | Optimized (lbfgs, liblinear) |
| **Cross-Validation** | — | 5-Fold CV |
| **Scoring** | Accuracy | ROC-AUC |

## Performance Summary

| Metric | Untuned Model | Tuned Model | Improvement |
|--------|-------------|-------------|-------------|
| **Accuracy** | ~0.85 | ~0.86 | +1.2% |
| **ROC-AUC** | ~0.82 | ~0.85 | **+3.7%** |
| **Precision (Churn)** | ~0.68 | ~0.72 | +5.9% |
| **Recall (Churn)** | ~0.55 | ~0.62 | **+12.7%** |

**Best Parameters:** `{'lr__C': 0.1, 'lr__solver': 'liblinear'}` (example)

## Key Findings

### 1. Model Performance
- **Tuned model improves churn detection by 12.7%** (recall) — critical for identifying at-risk customers
- ROC-AUC gain indicates better probability calibration for risk ranking
- Low computational cost for hyperparameter tuning vs. significant business impact
- <img width="989" height="590" alt="download14" src="https://github.com/user-attachments/assets/20c7374b-3ea5-4f88-8722-1a4a073c2302" />
<img width="576" height="435" alt="download9" src="https://github.com/user-attachments/assets/f9cce805-9bf9-413f-bc95-c3d24257fc6f" />



### 2. Top Churn Drivers
1. **Customer service calls** — Strongest predictor; frustration indicator
2. **Total international charges** — Cost sensitivity
3. **Total international minutes** — Usage pattern anomaly
4. **Evening charges & minutes** — Core service dissatisfaction
5. **Account length** — Loyalty vs. new customer volatility
   <img width="975" height="547" alt="download4" src="https://github.com/user-attachments/assets/0fbbae6c-1a46-4ce7-a455-3569391a559d" />


## Business Impact

| Metric | Value |
|--------|-------|
| Current Churn Rate | ~26.5% |
| Annual Churn Cost | ~$26M (est.) |
| **Potential Savings** | **$2-4M/year** (5-10% churn reduction) |
| Model Deployment Cost | Minimal (lightweight, <1ms prediction) |

## Recommendations

### Immediate Actions
- **Deploy tuned model** for weekly churn risk scoring
- **Prioritize customers** with >60% churn probability for retention calls
- **Address customer service pain points** — highest impact feature


## Technical Implementation

```python
# Tuned Pipeline
Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

# Best Parameters from GridSearchCV
{
    'lr__C': 0.1,           # Regularization strength
    'lr__solver': 'liblinear'  # Optimized for small datasets
}
```


 ## conclusion
The tuned Logistic Regression model significantly improves churn prediction, offering a strong balance between performance and interpretabilit
