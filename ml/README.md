# ML Training Module

This folder contains the machine learning training script for the Insurance Claims Analysis project.

## File

- `train_model.py` — trains a Random Forest model to classify customer insurance risk profiles.

## What the script does

The script:

1. Loads the processed insurance dataset
2. Creates a `Risk_Profile` target variable
3. Encodes categorical columns
4. Trains a Random Forest classifier
5. Prints model accuracy, classification report, confusion matrix, and feature importance
6. Saves trained model artifacts for later use in the dashboard:
   - `rf_model.pkl` — trained Random Forest model  
   - `encoders.pkl` — label encoders for categorical columns  
   - `features.pkl` — ordered feature list used by the model  
   - `feature_importance.csv` — feature importance results

## Dataset path

The script expects the processed dataset at:

```bash
../data/processed/fremtpl2_policy_claims.csv
