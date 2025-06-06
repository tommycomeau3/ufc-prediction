from scripts.preprocess import clean_ufc_data, scale_features
from scripts.model import train_and_evaluate_model

# Step 1: Load and clean the data
# Use the Kaggle “ufc-master.csv” file that ships with the repo
df_clean = clean_ufc_data('data/ufc-master.csv')
print("✅ Cleaned DataFrame shape:", df_clean.shape)
print(df_clean.head())

# Step 2: Scale and prepare features
X_scaled, y = scale_features(df_clean)
print("✅ X_scaled shape:", X_scaled.shape)
print("✅ y shape:", y.shape)
print("✅ First 5 target labels:", y.head().tolist())

# Step 3: Train and evaluate model
model = train_and_evaluate_model(X_scaled, y)
