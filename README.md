## Asset Trend Classification Using SVM

This project involved building a machine learning model to classify short-term asset return direction (up/down) using supervised learning techniques, with a focus on Support Vector Machines (SVM). The solution followed a full pipeline of feature engineering, preprocessing, modeling, and evaluation.

### Objective:
- Predict short-term return trend of a selected asset (binary: uptrend = 1, downtrend = 0).
- Focused on **binomial classification**, avoiding label formats like `[-1, 1]`.
- Constructed an SVM classifier with a custom feature set tailored to the chosen asset (e.g. equity, ETF, crypto).

### Feature Engineering:
- Generated features based on **price action and technical indicators**, such as:
  - Open - Close (`O - C`) and High - Low (`H - L`) intraday ranges
  - Lagged returns: $r_{t-1}, r_{t-2}, \dots$
  - Momentum: $P_t - P_{t-k}$
  - Simple Moving Average (SMA), Exponential Moving Average (EMA)
  - Return signs (positive/negative), and return magnitude thresholds
- Engineered thresholds to classify near-zero returns (< ±0.25%) as either noise or part of the majority class.

### Cost Function of Logistic Regression:

<!-- Logistic regression cost function rendered in white for dark mode -->
<br>

![Logistic Regression Cost](https://latex.codecogs.com/png.image?\dpi{120}&space;\color{White}J%28\theta%29%20%3D%20-%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5Em%20%5By%5E%7B%28i%29%7D%20\log%28h_%5Ctheta%28x%5E%7B%28i%29%7D%29%29%20%2B%20%281-y%5E%7B%28i%29%7D%29%20\log%281-h_%5Ctheta%28x%5E%7B%28i%29%7D%29%29%5D)

Where:
- $h_\theta(x)$ is the sigmoid activation function.
- $m$ is the number of training samples.
- The function penalizes incorrect predictions using log loss.

### Modeling Process:
- Used the **7-step ML pipeline**:
  1. Feature selection and data preprocessing
  2. Train/test split over a 5-year time horizon
  3. Standardization of features (z-score)
  4. Model training using **SVM classifier**
  5. Hyperparameter tuning using grid search (kernel, C, gamma)
  6. Model evaluation via:
     - Confusion matrix
     - Classification report (precision, recall, F1-score)
     - ROC AUC curve
  7. Interpretation of results

### Results & Evaluation:
- Achieved ROC AUC > 0.70 on out-of-sample data
- Demonstrated moderate predictive power on directional trends
- Explored trade-offs between overfitting and generalization through regularization and kernel choice
- Evaluated feature importance using recursive elimination and information gain

### Takeaways:
- Reinforced understanding of kernel methods, hyperparameter sensitivity, and data leakage risks in financial ML.
- Highlighted limitations of predicting short-term returns in efficient markets.
- Demonstrated full-stack ML implementation including model tuning, evaluation, and interpretation.
