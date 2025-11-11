import sys
import numpy as np
import pandas as pd
from joblib import load
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_regressor(out_dim: int = 4) -> keras.Model:
    model = Sequential([
        keras.Input(shape=(4,)),
        Dense(419, kernel_initializer='he_normal', activation='relu'),
        Dense(369, kernel_initializer= 'normal', activation='softmax'),
        Dense(150, kernel_initializer= 'he_normal', activation='linear'),
        Dense(out_dim, kernel_initializer='he_uniform', activation='linear')
        #Dense(128, activation="relu", kernel_initializer="he_normal"),
        #Dense(64,  activation="relu", kernel_initializer="he_normal"),
        #Dense(out_dim, activation="linear", kernel_initializer="he_uniform"),
    ])
    model.compile(optimizer=keras.optimizers.Adamax(2e-3),
                  loss="mse", metrics=["mae"])
    return model

def load_hits(hits_csv: str) -> pd.DataFrame:
    hitData = pd.read_csv(hits_csv, usecols=["SipmTime(ns)", "Sipm_Hit_XPosition", "Sipm_Hit_YPosition", "Sipm_Hit_ZPosition"])
    hitData["SipmTime(ns)"] = hitData["SipmTime(ns)"].astype(str).str.split().str[0].astype("float32")
    hitData = hitData.rename(columns={
        "Sipm_Hit_XPosition": "X(mm)",
        "Sipm_Hit_YPosition": "Y(mm)",
        "Sipm_Hit_ZPosition": "Z(mm)",
        "SipmTime(ns)": "Time(ns)",
    })
    return hitData

def main():
    if len(sys.argv) < 5:
        print("Usage: python train_DNN1.py sipmHits_output.csv annihilationEvents_output.csv neutronCaptureEvents_output.csv ann|neu")
        sys.exit(1)

    hits_csv, ann_csv, neu_csv, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].lower()
    if mode not in ("ann", "neu"):
        print("mode must be 'ann' or 'neu'")
        sys.exit(1)

    hits = load_hits(hits_csv)
    n = min(len(pd.read_csv(ann_csv)), len(pd.read_csv(neu_csv)))
    hits = hits.iloc[:n].reset_index(drop=True)

    sc_x = load("scaler_x.joblib")
    sc_y = load("scaler_y.joblib")
    sc_z = load("scaler_z.joblib")
    sc_t = load("scaler_t.joblib")

    X = hits[["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]].astype("float32").values
    X_s = np.column_stack([
        sc_x.transform(X[:, [0]]),
        sc_y.transform(X[:, [1]]),
        sc_z.transform(X[:, [2]]),
        sc_t.transform(X[:, [3]]),
    ]).astype("float32")

    if mode == "ann":
        weights = "weights_ann.weights.h5"
        y_scaler = load("y_scaler_ann.joblib")
        out_cols = ["xa", "ya", "za", "ta"]
        out_csv = "predictions_ann.csv"
    else:
        weights = "weights_neu.weights.h5"
        y_scaler = load("y_scaler_neu.joblib")
        out_cols = ["xn", "yn", "zn", "tn"]
        out_csv = "predictions_neu.csv"

    model = build_regressor(out_dim=4)
    model.load_weights(weights)

    Y_pred_s = model.predict(X_s, batch_size=512, verbose=0)
    Y_pred = y_scaler.inverse_transform(Y_pred_s)
    Y_pred[:, 3] = np.expm1(Y_pred[:, 3])
    Y_pred[:, 3] = np.clip(Y_pred[:, 3], 0.0, None) # avoid tiny negative times

    pd.DataFrame(Y_pred, columns=out_cols).to_csv(out_csv, index=False)
    print(f"Saved {mode} predictions to: {out_csv}")

if __name__ == "__main__":
    main()
