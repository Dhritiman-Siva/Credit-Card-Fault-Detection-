import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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
# RobustScaler is used as it is robust to extreme outliers
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])

# Drop unscaled original columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# 5. Display preview of processed dataset
print("\nCleaned & Normalized Data Preview:")
print(df[['scaled_time', 'scaled_amount', 'Class']].head())
print(f"\nFinal Shape: {df.shape}")

# 6. Save cleaned data to a separate compressed CSV (.zip)
output_file = 'cleaned_creditcard.csv.zip'
if not os.path.exists(output_file):
    df.to_csv(output_file, index=False, compression={'method': 'zip', 'archive_name': 'cleaned_creditcard.csv'})
    print(f"Cleaned dataset saved successfully to {output_file}")

# 7. Split into Train (80%), Validation (10%), and Test (10%)
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

# 8. Train Random Forest Model on 80% Train Set
print("\nTraining Random Forest Classifier on 80% Train set...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train, y_train)

# 9. Test the Model on the 10% Test Set
print("\n" + "="*50)
print("       TEST SET EVALUATION (10% Unseen Data)")
print("="*50)

test_preds = rf_model.predict(X_test)
test_probs = rf_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, test_preds)
tn, fp, fn, tp = cm.ravel()

print(f"ROC-AUC Score: {roc_auc_score(y_test, test_probs):.4f}")
print(f"Confusion Matrix:")
print(f"  True Negatives (Legit): {tn} | False Positives (False Alarms): {fp}")
print(f"  False Negatives (Missed): {fn}  | True Positives (Caught Fraud): {tp}\n")

print("Classification Report:")
print(classification_report(y_test, test_preds, digits=4, target_names=['Legitimate (0)', 'Fraud (1)']))

# Sample testing demonstration
sample_eval = pd.DataFrame({
    'Actual': y_test.iloc[:5].values,
    'Predicted': test_preds[:5],
    'Fraud_Probability': test_probs[:5].round(4)
})
print("Sample Test Predictions:")
print(sample_eval)
