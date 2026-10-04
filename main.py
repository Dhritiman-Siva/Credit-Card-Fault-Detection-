import os
import warnings
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# 1. Load dataset directly from zip
df = pd.read_csv('creditcard.csv.zip')
print(f"Initial shape: {df.shape}")

# 2. Handle missing values
missing_count = df.isnull().sum().sum()
print(f"Missing values found: {missing_count}")
df.dropna(inplace=True)

# 3. Remove duplicate transactions
df.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {df.shape}")

# 4. Normalize transaction data (Amount & Time)
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# 5. Save cleaned data to compressed CSV (.zip) if not already present
output_file = 'cleaned_creditcard.csv.zip'
if not os.path.exists(output_file):
    df.to_csv(output_file, index=False, compression={'method': 'zip', 'archive_name': 'cleaned_creditcard.csv'})
    print(f"Cleaned dataset saved successfully to {output_file}")

# 6. Split into Train (80%), Validation (10%), and Test (10%)
X = df.drop('Class', axis=1)
y = df['Class']

# First split: 80% train, 20% temp (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Second split: 10% validation, 10% test (stratified)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nData Splits (80-10-10):")
print(f"Train set:      {X_train.shape[0]} samples ({X_train.shape[0] / len(df):.0%})")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0] / len(df):.0%})")
print(f"Test set:       {X_test.shape[0]} samples ({X_test.shape[0] / len(df):.0%})")

# 7. Apply SMOTE Oversampling ONLY on 80% Training Data
print("\nApplying SMOTE oversampling to balance minority fraud cases in training data...")
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Train set shape before SMOTE: {X_train.shape[0]} | After SMOTE: {X_train_res.shape[0]}")

# 8. Train Random Forest Model
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)

# 9. Validation Set Evaluation (10% set for cross-validation / model tuning)
print("\n" + "="*50)
print("     VALIDATION SET EVALUATION (10% Tuning Data)")
print("="*50)
val_preds = rf_model.predict(X_val)
val_probs = rf_model.predict_proba(X_val)[:, 1]
print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")
print("Validation Classification Report:")
print(classification_report(y_val, val_preds, digits=4, target_names=['Legitimate (0)', 'Fraud (1)']))

# 10. Test Set Evaluation (10% final unseen test data)
print("="*50)
print("       TEST SET EVALUATION (10% Final Unseen)")
print("="*50)
test_preds = rf_model.predict(X_test)
test_probs = rf_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, test_preds)
tn, fp, fn, tp = cm.ravel()
print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
print(f"Confusion Matrix:")
print(f"  Legitimate: [True Negatives: {tn} | False Alarms: {fp}]")
print(f"  Fraud:      [Missed Fraud:   {fn} | Caught Fraud: {tp}]\n")
print("Test Classification Report:")
print(classification_report(y_test, test_preds, digits=4, target_names=['Legitimate (0)', 'Fraud (1)']))
