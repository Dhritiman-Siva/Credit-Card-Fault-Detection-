import pandas as pd
from sklearn.preprocessing import RobustScaler

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

# 6. Save cleaned data to a separate CSV
output_file = 'cleaned_creditcard.csv'
df.to_csv(output_file, index=False)
print(f"Cleaned dataset saved successfully to {output_file}")
