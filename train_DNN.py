#train_DNN.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from joblib import dump

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Masking, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from scikeras.wrappers import KerasRegressor
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def load_labels(ann_csv, neu_csv, hitData):
    ann = pd.read_csv(ann_csv, usecols=["Time(ns)", "X(mm)", "Y(mm)", "Z(mm)"])
    neu = pd.read_csv(neu_csv, usecols=["Time(ns)", "X(mm)", "Y(mm)", "Z(mm)"])
    n = min(len(ann), len(neu))
    print("n:", n)
    print("neu:", neu)
    ann = ann.iloc[:n].reset_index(drop=True)
    print("ann:", ann)
    neu = neu.iloc[:n].reset_index(drop=True)
    print("neu:", neu)
    # Y = [xa,ya,za,ta,  xn,yn,zn,tn]
    Y = np.hstack([
        ann[["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]].values,
        neu[["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]].values
    ]).astype("float32")

    X = hitData.iloc[:n]
    return Y, X



#def build_model(num_features: int, max_hits: int) -> keras.Model:
    #model = Sequential([
        #keras.Input(shape=(num_features, max_hits)),
        #Masking(mask_value=0.0),
        #Flatten(),
        #Dense(419, kernel_initializer="he_normal", activation="relu"),
        #Dense(369, kernel_initializer="he_normal", activation="relu"),   
        #Dense(150, kernel_initializer="he_normal", activation="linear"),
        #Dense(8,   kernel_initializer="he_uniform", activation="linear") 
    #])
    #model.compile(optimizer=keras.optimizers.Adamax(2e-3), loss="mse", metrics=["mae"])
    #return model
def build_model() -> keras.Model:
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
    

    hits_csv = sys.argv[1]
    ann_csv  = sys.argv[2]
    neu_csv  = sys.argv[3]
    
    hitfile = open(str(hits_csv))
    hitData = pd.read_csv(hitfile, usecols=["SipmTime(ns)", "Sipm_Hit_XPosition", "Sipm_Hit_YPosition", "Sipm_Hit_ZPosition"])
    print("hitData:", hitData)
    print("Time:", hitData["SipmTime(ns)"])
    print(hitData["SipmTime(ns)"].str.split(" "))
    print(hitData["SipmTime(ns)"].str.split(" ")[:,][0])
    hitData["SipmTime(ns)"] = ( hitData["SipmTime(ns)"].astype(str).str.split().str[0].astype("float32"))
    hitData = hitData.rename(columns={"Sipm_Hit_XPosition": "X(mm)", "Sipm_Hit_YPosition": "Y(mm)",
                                     "Sipm_Hit_ZPosition": "Z(mm)", "SipmTime(ns)": "Time(ns)"})
    #hitData = hitData.drop(columns=["SipmWavelength"], errors='ignore')
    print("hitData:", hitData)

    
    Y, X = load_labels(ann_csv, neu_csv, hitData)
    print("Y:", Y)
    print("Y.shape:", Y.shape)
    n_events = Y.shape[0]
    print("n_events", n_events)
    #hitdata = hitdata[0].str.split(',',expand=True).fillna('nan')
    #print("hitdata: ",hitdata )
    #X =  hitData.values
    print("X:", X)
    print("X.shape: ", X.shape)

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) #random_state=42)
    #num_hits = len(hitData.columns)
    #print("X_train:", X_train)
    #print("num_hits:", num_hits )
    #print("len(X):", int(len(X)))
    #print("len(X_train):", int(len(X_train)))
    #print("Y_train:", Y_train)

    #print(f"[info] X shape={X.shape}  (N={n_events}, features=4 [t,x,y,z], max_hits={max_hits})")
    #print(f"[info] Y shape={Y.shape}  (xa,ya,za,ta,xn,yn,zn,tn)")
    
    
    sc_x = MinMaxScaler(feature_range=(0, 1))
    sc_y   = MinMaxScaler(feature_range=(0, 1))
    sc_z   = MinMaxScaler(feature_range=(0, 1))
    sc_t   = MinMaxScaler(feature_range=(0, 1))

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    Y_s = y_scaler.fit_transform(Y).astype("float32")
    print("y_scaler mins:", y_scaler.min_)
    print("y_scaler scale:", y_scaler.scale_)
    print("y_scaler data_min_:", getattr(y_scaler, "data_min_", None))
    print("y_scaler data_max_:", getattr(y_scaler, "data_max_", None))
    #sc_y   = MinMaxScaler(feature_range=(0, 1))
    X_s = np.column_stack([ sc_x.fit_transform(X[["X(mm)"]]), sc_y.fit_transform(X[["Y(mm)"]]),
                          sc_z.fit_transform(X[["Z(mm)"]]), sc_t.fit_transform(X[["Time(ns)"]]), ]).astype("float32")               
    print("X_s: ", X_s)
    #Y_s = sc_y.fit_transform(Y)
    #print("Y_s: ", Y_s)
    num_features = 4
    est = KerasRegressor(
        model=build_model,
        #model__num_features=num_features,
        #model__max_hits=max_hits,
        epochs=80,
        batch_size=12,
        verbose=0,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=150)

    #scores = cross_val_score(est, X_s, Y_s, cv=cv)
    scores = cross_val_score(est, X_s, Y_s, cv=cv)
    print('Results: %.2f (%.2f) MSE' % (scores.mean(), scores.std()))

    #X_tr, X_val, Y_tr, Y_val = train_test_split(X_s, Y_s, test_size=0.2, random_state=150)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_s, Y_s, test_size=0.2, random_state=150)
    
    print("X_train:", X_tr)
    print("Y_train:", Y_tr)
    print("X_val:", X_val)
    print("Y_val:", Y_val)
    os.makedirs("./logs/train", exist_ok=True)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/train', histogram_freq=1)
    filepath = 'weights_out.weights.h5'
    #model = build_model(num_features=num_features, max_hits=max_hits)
    model = build_model()
    earlystopping = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    callbacks_list = [checkpoint, tbCallBack, earlystopping]
    history = model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=12,
        callbacks=callbacks_list,
        verbose=2
    )

    
    #model.save_weights("weights_final.weights.h5")
    model.save_weights("weights_out.weights.h5")


    #dump(sc_t,   "scaler_t.joblib")
    #dump(sc_xyz, "scaler_xyz.joblib")
    #dump(sc_y,   "scaler_y.joblib")
    dump(sc_x, "scaler_x.joblib")
    dump(sc_y, "scaler_y.joblib")
    dump(sc_z, "scaler_z.joblib")
    dump(sc_t, "scaler_t.joblib")
    dump(y_scaler, "scaler_y_targets.joblib")
    print(f"Saved best weights to: {filepath}")
    print(f"Saved scalers: scaler_t.joblib, scaler_x.joblib, scaler_y.joblib, scaler_z.joblib")


if __name__ == "__main__":
    main()
