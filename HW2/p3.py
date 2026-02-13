import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import matplotlib
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "training_dataset.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# not to be used as inputs
power_cols = ["total_watts", "w_big", "w_little", "w_gpu", "w_mem"]

feature_cols = [c for c in df.columns if c not in power_cols]

temp_cols = ["temp4", "temp5", "temp6", "temp7", "temp_gpu"]

targets = ["temp4", "temp5", "temp6", "temp7"]

print("Using feature columns:", feature_cols)

models_info = {}

X_all = df[feature_cols].copy()
if "freq_big_cluster" in X_all.columns:
    X_all["freq_big_cluster"] = X_all["freq_big_cluster"] / 1e9

for target in targets:
    print(f"\nTraining model for target: {target}")

    y = df[target].shift(-1)

    valid_idx = y.dropna().index
    X = X_all.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    non_temp_features = [c for c in X.columns if c not in temp_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), non_temp_features),
            ("pass", "passthrough", [c for c in X.columns if c in temp_cols])
        ]
    )

    mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                       solver='adam', max_iter=500, random_state=42,
                       early_stopping=True)

    pipeline = make_pipeline(preprocessor, mlp)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.4f}, R2: {r2:.4f}")

    model_path = os.path.join(OUT_DIR, f"mlp_{target}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Saved model to: {model_path}")

    models_info[target] = {"mse": float(mse), "r2": float(r2), "path": model_path}


with open(os.path.join(OUT_DIR, "models_summary.json"), "w") as f:
    json.dump(models_info, f, indent=2)

print("\nAll models trained and saved.")

matplotlib.use('Agg')

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def evaluate_on_test(csv_path, models_info, plot_name):
    df_test = pd.read_csv(csv_path)
    X_test = df_test[feature_cols].copy()
    if "freq_big_cluster" in X_test.columns:
        X_test["freq_big_cluster"] = X_test["freq_big_cluster"] / 1e9

    results = {}
    for t in targets:
        model_path = models_info[t]["path"]
        model = joblib.load(model_path)

        y_true = df_test[t].shift(-1).dropna().reset_index(drop=True)
        X_pred = X_test.loc[: len(y_true)-1].reset_index(drop=True)

        y_pred = model.predict(X_pred)

        mse = mean_squared_error(y_true, y_pred)
        results[t] = {"mse": float(mse)}

        print(f"{os.path.basename(csv_path)} - {t} MSE: {mse:.4f}")

        if t == 'temp4':
            time = np.arange(len(y_true))  
            plt.figure(figsize=(8,3))
            plt.plot(time, y_true, label='true')
            plt.plot(time, y_pred, label='pred')
            plt.xlabel('time [s]')
            plt.ylabel('temp4')
            plt.title(f"{plot_name} temp4: true vs pred")
            plt.legend()
            outpng = os.path.join(PLOT_DIR, f"{plot_name}_temp4.png")
            plt.tight_layout()
            plt.savefig(outpng)
            plt.close()
            print(f"Saved plot: {outpng}")

    return results

black_res = evaluate_on_test(os.path.join(os.path.dirname(__file__), 'testing_blackscholes.csv'), models_info, 'blackscholes')
body_res = evaluate_on_test(os.path.join(os.path.dirname(__file__), 'testing_bodytrack.csv'), models_info, 'bodytrack')

with open(os.path.join(OUT_DIR, 'test_results.json'), 'w') as f:
    json.dump({'blackscholes': black_res, 'bodytrack': body_res}, f, indent=2)

print('Evaluation complete. Test results saved to', os.path.join(OUT_DIR, 'test_results.json'))
