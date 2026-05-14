import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """
    Step 1: Load cleaned dataset
    """
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def prepare_features_and_target(df):
    """
    Step 2 & 3: Select useful ML features and Encode target labels
    """
    print("Preparing features and target variable...")
    
    # Step 2: Select useful ML features we created during preprocessing
    # Using the normalized/scaled versions of numeric columns where available
    feature_cols = [
        'route_id_encoded', 'hour', 'day_of_week', 'is_peak_hour', 
        'weather_severity', 'accident_impact', 'traffic_volume_scaled', 
        'average_speed_kmph_scaled', 'distance_km_scaled', 
        'base_eta_mins_scaled', 'temperature_celsius_scaled', 
        'precipitation_mm_scaled', 'visibility_km_scaled', 
        'toll_fee_inr_scaled', 'surge_pricing_active'
    ]
    
    # We only keep columns that actually exist in the dataframe to prevent errors
    actual_features = [col for col in feature_cols if col in df.columns]
    X = df[actual_features]
    
    # Step 3: Encode target labels: Low, Medium, High
    # We fit a LabelEncoder on the original 'congestion_level' string column
    # so we can save it and reuse it later to decode predictions back to words.
    le = LabelEncoder()
    y = le.fit_transform(df['congestion_level'])
    
    return X, y, le, actual_features

def split_data(X, y):
    """
    Step 4: Split data into Training and Testing sets
    """
    print("Splitting data into 80% training and 20% testing...")
    # random_state=42 ensures reproducibility (we get the exact same split every time we run it)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Step 5: Train Random Forest Classifier
    """
    print("Training Random Forest Classifier...")
    # Random Forest is a robust algorithm that builds multiple decision trees
    # n_estimators=100 means we are building 100 trees
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test, target_names, feature_names):
    """
    Step 6: Show accuracy, classification report, confusion matrix, and feature importance
    """
    print("\n--- MODEL EVALUATION ---")
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # 1. Accuracy
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc * 100:.2f}%\n")
    
    # 2. Classification Report
    print("Classification Report:")
    # target_names allows the report to show 'Low', 'Medium', 'High' instead of numbers
    print(classification_report(y_test, predictions, target_names=target_names))
    
    # 3. Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    # 4. Feature Importance
    print("\nFeature Importances (Top 5):")
    importances = model.feature_importances_
    
    # Combine feature names and their importance scores
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    # Sort them to show the most important features at the top
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(5)
    print(feat_imp.to_string(index=False))

def save_model(model, label_encoder, models_dir):
    """
    Step 7 & 8: Save trained model and label encoder
    """
    print("\nSaving model and encoder...")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'congestion_model.pkl')
    le_path = os.path.join(models_dir, 'label_encoder.pkl')
    
    # joblib is great for saving scikit-learn models efficiently
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, le_path)
    
    print(f"Model successfully saved to: {model_path}")
    print(f"Label Encoder successfully saved to: {le_path}")

def main():
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'datasets', 'cleaned_traffic_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    # 1. Load data
    df = load_data(data_path)
    
    # 2 & 3. Prepare features and target
    X, y, label_encoder, feature_names = prepare_features_and_target(df)
    
    # Get the names of the classes (e.g., ['High', 'Low', 'Medium'])
    target_names = label_encoder.classes_
    
    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 5. Train model
    model = train_model(X_train, y_train)
    
    # 6. Evaluate model
    evaluate_model(model, X_test, y_test, target_names, feature_names)
    
    # 7 & 8. Save model and encoder
    save_model(model, label_encoder, models_dir)

if __name__ == "__main__":
    main()
