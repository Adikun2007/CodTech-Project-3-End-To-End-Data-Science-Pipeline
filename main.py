import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, RocCurveDisplay, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots


# 1. Load the dataset and split into train/test sets (stratified)
df = pd.read_csv('creditcard.csv')

features = df.drop('Class', axis=1) # features are all columns except 'Class'
target = df['Class'] # target is the 'Class' column

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target)

# 2. Scale 'Amount' and 'Time' features
scaler = StandardScaler()
x_train[['Amount', 'Time']] = scaler.fit_transform(x_train[['Amount', 'Time']])
x_test[['Amount', 'Time']] = scaler.transform(x_test[['Amount', 'Time']]) # only transform on test set Because in real life when your deployed model gets a new transaction, it won't know the mean/std of all future data. It only knows what it learned from training. So we simulate that here

# 3. Apply SMOTE to the training data only
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train,y_train)

# Trying different model and saving the best one
# 1. Logistic Regression
# log_reg = LogisticRegression(max_iter=500, random_state=42)
# log_reg.fit(x_train_smote, y_train_smote)
# log_reg_proba = log_reg.predict_proba(x_test)[:, 1] # gives us the probability of the positive class (fraud) for each test sample, which we can use for thresholding later on. If we just did log_reg.predict(x_test), it would give us the predicted class labels (0 or 1) based on the default threshold of 0.5, which is not what we want at this stage since we want to optimize the threshold later.
# log_reg_predict = log_reg.predict(x_test)
# print("=" * 25)
# print("Logistic Regression Performance:")
# print(classification_report(y_test, log_reg_predict, target_names=['Legit', 'Fraud']))
# print(f"ROC-AUC Score: {roc_auc_score(y_test, log_reg_proba):.4f}")
# print("=" * 25)

# # 2. Random Forest
# rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_clf.fit(x_train_smote, y_train_smote)
# rf_proba = rf_clf.predict_proba(x_test)[:, 1]
# rf_predict = rf_clf.predict(x_test)
# print("=" * 25)
# print("Random Forest Performance:")
# print(classification_report(y_test, rf_predict, target_names=['Legit', 'Fraud']))
# print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_proba):.4f}")
# print("=" * 25)

# 3. XGBoost
xgb_clf = XGBClassifier(n_estimators=500, random_state=42, eval_metric='logloss', n_jobs=-1)
xgb_clf.fit(x_train_smote, y_train_smote)
xgb_proba = xgb_clf.predict_proba(x_test)[:, 1] # gives us the probability of the positive class (fraud) for each test sample, which we can use for thresholding later on. If we just did log_reg.predict(x_test), it would give us the predicted class labels (0 or 1) based on the default threshold of 0.5, which is not what we want at this stage since we want to optimize the threshold later.
xgb_predict = xgb_clf.predict(x_test)
print("=" * 25)
print("XGBoost Performance:")
print(classification_report(y_test, xgb_predict, target_names=['Legit', 'Fraud']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_proba):.4f}")
print("=" * 25)

#  i have tested 3 models and analyzed their performance. XGBoost is the best performing model for 500 estimators, so I will save it for tuning and threshold optimization and comment out the other models, you can try different models and use the best one.

# 4. Hyperparameter tuning the XGBoost model
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3,4,5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1], # Subsample is a regularization technique that helps prevent overfitting by randomly sampling a fraction of the training data for each boosting round. A value of 0.8 means that each tree will be trained on 80% of the data, which can help improve generalization performance, especially in cases where the model might otherwise overfit to the training data. You can experiment with different values (e.g., 0.6, 0.7) to see which gives the best performance on your validation set.
    'colsample_bytree': [0.8, 1], # Similar to subsample, colsample_bytree is another regularization technique, especially when you have a large number of features. You can experiment with different values (e.g., 0.6, 0.7) to see which gives the best performance on your validation set.
    'min_child_weight': [1, 5, 10], 
    'scale_pos_weight': [1] # Because smote already balances the classes, this is important for imbalanced datasets like ours, it helps the model pay more attention to the minority class (fraud) during training. You can experiment with different values to see which gives the best performance.
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    random_state=42
)
search.fit(x_train_smote, y_train_smote)

print(f"\nBest params: {search.best_params_}")
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

best_xgb = search.best_estimator_

# 5. Threshold optimization
tuned_prob = best_xgb.predict_proba(x_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, tuned_prob)

# Strategy: maximise F1 but enforce minimum fraud recall of 0.80
# In fraud detection, missing a fraud (false negative) costs more than a false alarm (legit transaction flagged as fraud). So we want to find the threshold that gives us the best F1 score while ensuring that we catch at least 80% of the fraud cases (recall >= 0.80).

f1_scores = np.where(
    recalls[:-1] >= 0.80,
    2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-6), # add small epsilon to avoid division by zero
    0   
)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print(f"\nOptimal threshold:  {best_threshold:.4f}")
print(f"At this threshold:")
print(f"  Precision (fraud): {precisions[best_index]:.4f}")
print(f"  Recall    (fraud): {recalls[best_index]:.4f}")
print(f"  F1        (fraud): {f1_scores[best_index]:.4f}")

# Apply the best threshold to get final predictions
final_preds = (tuned_prob >= best_threshold).astype(int)
print("\n--- Tuned XGBoost Results ---")
print(classification_report(y_test, final_preds, target_names=['Legit', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, tuned_prob):.4f}")

# 6. Save the best model and scaler
joblib.dump(best_xgb, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_threshold, 'best_threshold.pkl') # Save the best threshold for use in deployment

print("\nSaved: fraud_detection_model.pkl")
print("Saved: scaler.pkl")
print("Saved: best_threshold.pkl")
print("Model, scaler, and threshold saved successfully!")

# 7. Plots
fig, axes = plt.subplots(2,2, figsize=(14,11))
fig.suptitle("XGBoost Fraud Detection - Results", fontsize=16, fontweight='bold', y=0.95)

# --- Confusion Matrix ---
ax = axes[0, 0]
cm = confusion_matrix(y_test, final_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Legit', 'Fraud'],
            yticklabels=['Legit', 'Fraud'])
ax.set_title('Confusion Matrix')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
 
# --- Precision-Recall Curve ---
ax = axes[0, 1]
ax.plot(recalls[:-1], precisions[:-1], color='steelblue', label='PR Curve')
ax.axvline(recalls[best_index], color='red', linestyle='--',
           label=f'Best Threshold = {best_threshold:.2f}')
ax.set_title('Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend()
ax.grid(True, alpha=0.3)
 
# --- ROC Curve ---
ax = axes[1, 0]
RocCurveDisplay.from_predictions(y_test, tuned_prob, ax=ax, color='steelblue')
ax.plot([0, 1], [0, 1], 'k--')  # random classifier baseline
ax.set_title('ROC Curve')
ax.grid(True, alpha=0.3)
 
# --- Feature Importance ---
ax = axes[1, 1]
importance_df = pd.DataFrame({
    'feature': features.columns,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=True).tail(20)
 
colors = ['tomato' if f in ['Amount', 'Time'] else 'steelblue' for f in importance_df['feature']]
ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
ax.set_title('Top 20 Feature Importances')
ax.set_xlabel('Importance Score')
ax.grid(True, axis='x', alpha=0.3)
 
plt.tight_layout()
plt.savefig('plots_results.png', dpi=150, bbox_inches='tight')
print("Saved: plots_results.png")
 