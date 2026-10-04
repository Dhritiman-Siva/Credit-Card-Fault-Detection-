import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

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
df.to_csv(output_file, index=False, compression={'method': 'zip', 'archive_name': 'cleaned_creditcard.csv'})
print(f"Cleaned dataset saved successfully to {output_file}")

# 7. Split into Train (80%), Validation (10%), and Test (10%)
X = df.drop('Class', axis=1)
y = df['Class']

# First split: 80% train, 20% temp (stratified to preserve fraud ratio)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Second split: 10% validation, 10% test (50% of the 20% temp)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nData Splits (80-10-10):")
print(f"Train set:      {X_train.shape[0]} samples ({X_train.shape[0] / len(df):.0%})")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0] / len(df):.0%})")
print(f"Test set:       {X_test.shape[0]} samples ({X_test.shape[0] / len(df):.0%})")
