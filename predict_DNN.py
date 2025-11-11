#predict_DNN.py
import sys
import numpy as np
import pandas as pd
from joblib import load
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Masking, Flatten, Dense


def load_truth(ann_csv, neu_csv):
    
    if ann_csv is None or neu_csv is None:
        return None

    ann = pd.read_csv(ann_csv, usecols=["Time(ns)","X(mm)","Y(mm)","Z(mm)"])
    neu = pd.read_csv(neu_csv,  usecols=["Time(ns)","X(mm)","Y(mm)","Z(mm)"])
    n_events = min(len(ann), len(neu))
    Y_true = np.hstack([
        ann[["X(mm)","Y(mm)","Z(mm)","Time(ns)"]].values[:n_events],
        neu[["X(mm)","Y(mm)","Z(mm)","Time(ns)"]].values[:n_events]
    ]).astype("float32")
    return Y_true

def build_model() -> keras.Model:
    
    #model = Sequential([
        #keras.Input(shape=(num_features, max_hits)),
        #Masking(mask_value=0.0),
        #Flatten(),
        #Dense(419, activation="relu", kernel_initializer="he_normal"),
        #Dense(369, activation="relu", kernel_initializer="he_normal"),   
        #Dense(150, activation="linear", kernel_initializer="he_normal"),
        #Dense(8,   activation="linear", kernel_initializer="he_uniform"),  
    #])
    #return model

    model = Sequential([
        keras.Input(shape=(4,)),
        Dense(128, activation="relu", kernel_initializer="he_normal"),
        Dense(64,  activation="relu", kernel_initializer="he_normal"),
        #Dense(8,   activation="linear", kernel_initializer="he_uniform"),
        Dense(8,   activation="tanh",  kernel_initializer="he_uniform"),
    ])
    model.compile(optimizer=keras.optimizers.Adamax(2e-3), loss="mse", metrics=["mae"])
    return model

def main():
    
    hits_csv   = sys.argv[1]
    
    ann_csv    = sys.argv[2] 
    neu_csv    = sys.argv[3] 
    preds_out = "predictions_DNN.csv"
    
    ann = pd.read_csv(ann_csv, usecols=["Time(ns)","X(mm)","Y(mm)","Z(mm)"])
    neu = pd.read_csv(neu_csv,  usecols=["Time(ns)","X(mm)","Y(mm)","Z(mm)"])
    #hits = pd.read_csv(hits_csv, usecols=["Time(ns)","X(mm)","Y(mm)","Z(mm)"])

    hitfile = open(str(hits_csv))
    hitData = pd.read_csv(hitfile, usecols=["SipmTime(ns)", "Sipm_Hit_XPosition", "Sipm_Hit_YPosition", "Sipm_Hit_ZPosition"])
    print("hitData:", hitData)
    print("Time:", hitData["SipmTime(ns)"])
    print(hitData["SipmTime(ns)"].str.split(" "))
    print(hitData["SipmTime(ns)"].str.split(" ")[:,][0])
    hitData["SipmTime(ns)"] = ( hitData["SipmTime(ns)"].astype(str).str.split().str[0].astype("float32"))
    hits = hitData.rename(columns={"Sipm_Hit_XPosition": "X(mm)", "Sipm_Hit_YPosition": "Y(mm)",
                                     "Sipm_Hit_ZPosition": "Z(mm)", "SipmTime(ns)": "Time(ns)"})
    #hitData = hitData.drop(columns=["SipmWavelength"], errors='ignore')
    #print("hitData:", hits)

    print('ann', ann)
    print('neu', neu)
    print('hits', hits)
    n_events = min(len(ann), len(neu))
    print('n_events', n_events)

   #X, max_hits = load_hits_build_X(hits_csv, n_events)
    #print("X:", X)
    #print("max_hits", max_hits)
    X = hits.iloc[:n_events]
    

    #sc_t   = load("scaler_t.joblib")
    #print("sc_t", sc_t)
    #sc_xyz = load("scaler_xyz.joblib")
    #sc_y   = load("scaler_y.joblib")

    sc_x = load("scaler_x.joblib")
    sc_y   = load("scaler_y.joblib")
    sc_z   = load("scaler_z.joblib")
    sc_t   = load("scaler_t.joblib")
    y_scaler = load("scaler_y_targets.joblib")

    #X_s = np.empty_like(X)
    #X_s[:, 0, :] = sc_t.transform(X[:, 0, :])     
    #X_s[:, 1, :] = sc_xyz.transform(X[:, 1, :])   
    #X_s[:, 2, :] = sc_xyz.transform(X[:, 2, :])   
    #X_s[:, 3, :] = sc_xyz.transform(X[:,  3, :]) 
    #X_s = np.nan_to_num(X_s, nan=0.0)
    #X_s = np.column_stack([ sc_x.fit_transform(X[["X(mm)"]]), sc_y.fit_transform(X[["Y(mm)"]]),
                          #sc_z.fit_transform(X[["Z(mm)"]]), sc_t.fit_transform(X[["Time(ns)"]]), ]).astype("float32")
    X_s = np.column_stack([ sc_x.transform(X[["X(mm)"]]), sc_y.transform(X[["Y(mm)"]]),
                          sc_z.transform(X[["Z(mm)"]]), sc_t.transform(X[["Time(ns)"]]), ]).astype("float32") 
    print("X_s", X_s)
    num_features = 4
    #model = build_model(num_features=num_features, max_hits=max_hits)
    model = build_model()

    #model.save_weights("weights_out.weights.h5")
    weights_in = "weights_out.weights.h5"
    model.load_weights(weights_in)

    #model.load_weights(weights_in)

    #Y_pred_s = model.predict(X_s, batch_size=12, verbose=0)      
    #Y_pred   = y_scaler.inverse_transform(Y_pred_s)

    Y_pred_s = model.predict(X_s, batch_size=256, verbose=0)
    Y_pred_s = np.clip(Y_pred_s, -1.0, 1.0)
    Y_pred   = y_scaler.inverse_transform(Y_pred_s)

    cols = ["xa","ya","za","ta","xn","yn","zn","tn"]
    pd.DataFrame(Y_pred, columns=cols).to_csv(preds_out, index=False)
    print("Predictions saved to:", preds_out)

    Y_true = load_truth(ann_csv, neu_csv)
    if Y_true is not None and Y_true.shape == Y_pred.shape:
        mse  = np.mean((Y_pred - Y_true)**2, axis=0)
        rmse = np.sqrt(mse)
        print("RMSE [xa ya za ta xn yn zn tn]:", np.round(rmse, 3))


if __name__ == "__main__":
    main()
