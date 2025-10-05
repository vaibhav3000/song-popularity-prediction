import pandas as pd
import numpy as np
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

def train_model(model, X_train, y_train, X_test_data, folds):
    """
    Train a model using cross-validation and return out-of-fold and test predictions.

    Args:
        model: The machine learning model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test_data (pd.DataFrame): Test features.
        folds: StratifiedKFold object.

    Returns:
        tuple: A tuple containing out-of-fold predictions and test predictions.
    """
    out_of_fold_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test_data))
    model_name = model.__class__.__name__

    for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Model-specific training with early stopping
        if isinstance(model, lgb.LGBMClassifier):
            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        elif isinstance(model, cb.CatBoostClassifier):
            model.fit(X_train_fold, y_train_fold,
                      eval_set=[(X_val_fold, y_val_fold)],
                      early_stopping_rounds=100,
                      verbose=0)
        else: # For XGBoost and others without a callback in fit
            model.fit(X_train_fold, y_train_fold)

        out_of_fold_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
        test_preds += model.predict_proba(X_test_data)[:, 1] / folds.n_splits

    auc_score = roc_auc_score(y_train, out_of_fold_preds)
    print(f"Local CV AUC for {model_name}: {auc_score:.5f}")
    return out_of_fold_preds, test_preds

def create_features(df):
    """Engineer new features for the model."""
    df['danceability_X_energy'] = df['danceability'] * df['energy']
    df['loudness_X_energy'] = df['loudness'] * df['energy']
    
    # Interaction features based on key, time signature, and mode
    for feature in ['key', 'time_signature', 'audio_mode']:
        for stat in ['mean', 'std']:
            grouped = df.groupby(feature)['danceability'].agg(stat).rename(f'{feature}_danceability_{stat}')
            df = df.merge(grouped, on=feature, how='left')
    return df

def main():
    """Main function to run the entire ML pipeline."""
    try:
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        submission_ids = test_df['id']
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found. Make sure they are in the correct directory.")
        return

    # Combine for consistent feature engineering
    combined_df = pd.concat([train_df.drop('song_popularity', axis=1), test_df], ignore_index=True)
    combined_df = create_features(combined_df)

    # Separate back into train and test
    train_processed = combined_df.iloc[:len(train_df)].copy()
    test_processed = combined_df.iloc[len(train_df):].copy()
    train_processed['song_popularity'] = train_df['song_popularity']

    features = [col for col in train_processed.columns if col not in ['id', 'song_popularity']]
    X = train_processed[features]
    y = train_processed['song_popularity']
    X_test = test_processed[features]

    # Preprocessing: Imputation and Scaling
    imputer = IterativeImputer(max_iter=10, random_state=42)
    X = pd.DataFrame(imputer.fit_transform(X), columns=features)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=features)
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=features)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=features)
    
    # Model Training
    N_SPLITS = 10
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    model_predictions = {}

    # LightGBM Model
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'n_estimators': 2000,
        'learning_rate': 0.01, 'num_leaves': 20, 'max_depth': 3,
        'seed': 42, 'n_jobs': -1, 'verbose': -1,
        'colsample_bytree': 0.7, 'subsample': 0.7
    }
    _, model_predictions['lgbm'] = train_model(lgb.LGBMClassifier(**lgb_params), X, y, X_test, skf)
    
    # XGBoost Model
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'eta': 0.02,
        'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'seed': 42, 'n_estimators': 1000, 'use_label_encoder': False
    }
    _, model_predictions['xgb'] = train_model(xgb.XGBClassifier(**xgb_params), X, y, X_test, skf)

    # CatBoost Model
    cb_params = {
        'iterations': 2000, 'learning_rate': 0.02, 'depth': 4,
        'loss_function': 'Logloss', 'eval_metric': 'AUC',
        'random_seed': 42, 'verbose': 0, 'allow_writing_files': False
    }
    _, model_predictions['catboost'] = train_model(cb.CatBoostClassifier(**cb_params), X, y, X_test, skf)

    # Blending and Submission
    final_predictions = (model_predictions['lgbm'] + model_predictions['xgb'] + model_predictions['catboost']) / 3.0
    
    submission_df = pd.DataFrame({'id': submission_ids, 'song_popularity': final_predictions})
    submission_df.to_csv('Top.csv', index=False)
    
    print("Submission file 'Top.csv' created successfully!")
    print("\nSubmission Preview:")
    print(submission_df.head())

if __name__ == '__main__':
    main()