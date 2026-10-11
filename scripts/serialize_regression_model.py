import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

train_df = pd.read_csv('data/train_regression_preprocessed.csv')
test_df = pd.read_csv('data/test_regression_preprocessed.csv')

X_train = train_df.drop(columns=['koi_prad'])
y_train = train_df['koi_prad']

X_test = test_df.drop(columns=['koi_prad'])
y_test = test_df['koi_prad']

print("Training champion Gradient Boosting Regressor...")
gbr = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)
r2 = r2_score(y_test, y_pred)
import numpy as np
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = mean_absolute_error(y_test, y_pred)

print(f"Regression Champion R2:   {r2:.4f}")
print(f"Regression Champion RMSE: {rmse:.4f}")
print(f"Regression Champion MAE:  {mae:.4f}")

reg_artifact = {
    'model': gbr,
    'features': list(X_train.columns),
    'target': 'koi_prad',
    'metrics': {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae)
    }
}

os.makedirs('models', exist_ok=True)
joblib.dump(reg_artifact, 'models/best_regression_model.joblib')
print("Champion regression model serialized to models/best_regression_model.joblib.")
