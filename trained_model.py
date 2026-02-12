# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('alzheimers_disease_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check target distribution
print(f"\nDiagnosis distribution:")
print(df['Diagnosis'].value_counts())

# Handle the XXXConfid values - these appear to be placeholder/missing values
# Replace XXXConfid with NaN and then drop or fill
df = df.replace('XXXConfid', np.nan)

# Drop columns that are not useful for prediction
columns_to_drop = ['PatientID', 'DoctorInCharge', 'Ethnicity']  # Ethnicity might be sensitive
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Handle missing values - drop rows with missing target or important features
df = df.dropna(subset=['Diagnosis'])
df = df.fillna(df.median(numeric_only=True))

print(f"\nAfter cleaning, dataset shape: {df.shape}")

# Separate features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {len(numerical_cols)} columns")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print(f"\nTarget classes: {target_encoder.classes_}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nModel Performance:")
print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20 most important features:")
print(feature_importance.head(20))

# Save the model and encoders
joblib.dump(model, 'alzheimers_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')

print(f"\n✅ Model saved successfully!")
print(f"Model expects {len(X.columns)} features")