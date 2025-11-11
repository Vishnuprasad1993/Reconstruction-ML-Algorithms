import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
    model.compile(optimizer=keras.optimizers.Adamax(2e-3), loss="mse", metrics=["mae"])
    return model

def load_hits(hits_csv: str) -> pd.DataFrame:
    hitData = pd.read_csv(
        hits_csv,
        usecols=["SipmTime(ns)", "Sipm_Hit_XPosition", "Sipm_Hit_YPosition", "Sipm_Hit_ZPosition"]
    )
    hitData["SipmTime(ns)"] = hitData["SipmTime(ns)"].astype(str).str.split().str[0].astype("float32")
    hitData = hitData.rename(columns={
        "Sipm_Hit_XPosition": "X(mm)",
        "Sipm_Hit_YPosition": "Y(mm)",
        "Sipm_Hit_ZPosition": "Z(mm)",
        "SipmTime(ns)": "Time(ns)",
    })
    return hitData

def plot_input_distributions(X):
    cols = ["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]
    plt.figure(figsize=(12,8))
    for i, c in enumerate(cols):
        plt.subplot(2,2,i+1)
        plt.hist(X[:, i], bins=100, histtype='step', color='b')
        plt.title(f"Distribution of {c}")
        plt.xlabel(c)
        plt.ylabel("Counts")
    plt.savefig(f"sipmhits_histogram.png")
    plt.tight_layout()
    plt.show()

def plot_correlation(X):
    df = pd.DataFrame(X, columns=["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"])
    plt.figure(figsize=(6,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Feature Correlation (SiPM hits)")
    plt.savefig(f"correlation.png")
    plt.show()

def plot_scaling_comparison(X_raw, X_scaled):
    cols = ["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]
    X_raw_filtered = X_raw.copy()
    mask = X_raw[:, 3] > 50000
    print("mask", mask)
    if True in mask:
        print("True")
    X_raw_filtered = X_raw[mask]
    print("X_raw_filtered", X_raw_filtered)
    X_scaled_filtered = X_scaled[mask]
    for i, c in enumerate(cols):
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        plt.hist(X_raw[:, i], bins=100, histtype='step', color='b')
        #plt.hist(X_raw_filtered[:, i], bins=100, histtype='step', color='b')
        plt.title(f"Raw {c}")
        plt.subplot(1,2,2)
        plt.hist(X_scaled[:, i], bins=100, histtype='step', color='g')
        #plt.hist(X_scaled_filtered[:, i], bins=100, histtype='step', color='g')
        plt.title(f"Scaled {c}")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"sipmhits{c}_scaling.png")

#def load_labels(ann_csv: str, neu_csv: str, n: int) -> np.ndarray:
def load_labels(ann_csv: str, neu_csv: str) -> np.ndarray:
    #ann = pd.read_csv(ann_csv, usecols=["Time(ns)", "X(mm)", "Y(mm)", "Z(mm)"]).iloc[:n].reset_index(drop=True)
    #neu = pd.read_csv(neu_csv, usecols=["Time(ns)", "X(mm)", "Y(mm)", "Z(mm)"]).iloc[:n].reset_index(drop=True)
    Y_ann = pd.read_csv(ann_csv, usecols=["Time(ns)", "X(mm)", "Y(mm)", "Z(mm)"]).reset_index(drop=True)
    Y_neu = pd.read_csv(neu_csv, usecols=["Time(ns)", "X(mm)", "Y(mm)", "Z(mm)"]).reset_index(drop=True)
    
    #Y = np.hstack([
        #ann[["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]].values,
        #neu[["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]].values
    #]).astype("float32")
    return Y_ann, Y_neu

def plot_targets(Y_ann, Y_neu, mode = None):
    cols = ["x", "y", "z", "t"]
    plt.figure(figsize=(12,8))
    print("Y_ann_plot:", Y_ann)
    if mode == None:
      for i, c in enumerate(cols):
         plt.subplot(2,2,i+1)
         plt.hist(Y_ann[:, i], bins=100, alpha=0.5, label='Annihilation')
         plt.hist(Y_neu[:, i], bins=100, alpha=0.5, label='Neutron capture')
         plt.title(f"Target distribution {c}")
         plt.legend()
         plt.xlabel(c)
      plt.savefig(f"target_histogram.png")
      plt.tight_layout()
      plt.show()
    elif mode == "log":
        for i, c in enumerate(cols):
           plt.subplot(2,2,i+1)
           plt.hist(Y_ann[:, i], bins=100, alpha=0.5, label='Annihilation')
           plt.hist(Y_neu[:, i], bins=100, alpha=0.5, label='Neutron capture')
           plt.title(f"Log Scaled Target distribution {c}")
           plt.legend()
           plt.xlabel(c)
        plt.savefig(f"log scaled_target_histogram.png")
        plt.tight_layout()
        plt.show()
    elif mode == "minmaxscaler":
        for i, c in enumerate(cols):
           plt.subplot(2,2,i+1)
           plt.hist(Y_ann[:, i], bins=100, alpha=0.5, label='Annihilation')
           plt.hist(Y_neu[:, i], bins=100, alpha=0.5, label='Neutron capture')
           plt.title(f"Min Max Scaled Target distribution {c}")
           plt.legend()
           plt.xlabel(c)
        plt.savefig(f"min_max_scaled_target_histogram.png")
        plt.tight_layout()
        plt.show()


def main():
    if len(sys.argv) < 4:
        print("Usage: python train_DNN1.py sipmHits_output.csv annihilationEvents_output.csv neutronCaptureEvents_output.csv")
        sys.exit(1)

    hits_csv, ann_csv, neu_csv = sys.argv[1], sys.argv[2], sys.argv[3]

    hits = load_hits(hits_csv)
    a = hits[hits["Time(ns)"] > 50000]
    print("a", a)
    print("len(a)", len(a))
    print(hits["Time(ns)"] > 50000)
    #n = min(len(pd.read_csv(ann_csv)), len(pd.read_csv(neu_csv)))
    #hits = hits.iloc[:n].reset_index(drop=True)
    #Y = load_labels(ann_csv, neu_csv, n)  # (n, 8)

    Y_ann, Y_neu = load_labels(ann_csv, neu_csv)
    Y_ann = Y_ann[["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]].astype("float32").values
    Y_neu = Y_neu[["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]].astype("float32").values
    X = hits[["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]].astype("float32").values

    print("X:", X)
    print("Y_ann:", Y_ann)
    print("Y_neu:", Y_neu)
    print("X type:", type(X))
    X_tr_length = int(len(X) - 0.2 * len(X))
    X_tr = X[0: X_tr_length]
    X_val = X[X_tr_length: len(X)]
    print("X type:", type(X))
    print("X_tr:", X_tr)
    print("X_val:", X_val)
    print("len(X_tr):", len(X_tr))
    print("len(X_val):", len(X_val))
    
    Ya_tr_length = int(len(Y_ann) - 0.2 * len(Y_ann))
    Ya_tr = Y_ann[0: Ya_tr_length]
    Ya_val = Y_ann[Ya_tr_length: len(Y_ann)]

    Yn_tr_length = int(len(Y_neu) - 0.2 * len(Y_neu))
    Yn_tr = Y_neu[0: Yn_tr_length]
    Yn_val = Y_neu[Yn_tr_length: len(Y_neu)]

    print("Ya_tr:", Ya_tr)
    print("Ya_val:", Ya_val)
    print("len(Ya_tr):", len(Ya_tr))
    print("len(Ya_val):", len(Ya_val))

    print("Yn_tr:", Yn_tr)
    print("Yn_val:", Yn_val)
    print("len(Yn_tr):", len(Yn_tr))
    print("len(Yn_val):", len(Yn_val))

    mask = X[:, 3] > 50000          
    b = X[mask]
    print("b",b)
    print("len(b)", len(b))

    plot_input_distributions(X)
    plot_correlation(X)
    
    #Y_ann = Y[:, 0:4]  # xa,ya,za,ta
    #Y_neu = Y[:, 4:8]  # xn,yn,zn,tn'
    print("X:", X)
    print("Y_ann:", Y_ann)
    print("Y_neu:", Y_neu)
    print("len(X):", len(X))
    print("len(Y_ann):", len(Y_ann))
    print("len(Y_neu):", len(Y_neu))


    sc_x = MinMaxScaler((0, 1))
    sc_y = MinMaxScaler((0, 1))
    sc_z = MinMaxScaler((0, 1))
    sc_t = MinMaxScaler((0, 1))

    X_s = np.column_stack([
        sc_x.fit_transform(X_tr[:, [0]]),
        sc_y.fit_transform(X_tr[:, [1]]),
        sc_z.fit_transform(X_tr[:, [2]]),
        sc_t.fit_transform(X_tr[:, [3]]),
    ]).astype("float32")
    
    print("X_s:", X_s)

    plot_scaling_comparison(X_tr, X_s)
    #split into annihilation and neutroncapture targets (each is 4 vector)
    
    
    print("Y_ann:", Ya_tr)
    print("Y_neu:", Yn_tr)

    plot_targets(Ya_tr, Yn_tr)

    Ya_tr[:, 3] = np.log1p(Ya_tr[:, 3]) 
    Yn_tr[:, 3] = np.log1p(Yn_tr[:, 3]) 
    
    Y_ann_log = Ya_tr
    Y_neu_log = Yn_tr
    
    plot_targets(Y_ann_log, Y_neu_log, "log")
    
    print("Y_ann_log:", Y_ann_log)
    print("Y_neu_log:", Y_neu_log)


    y_scaler_ann = MinMaxScaler(feature_range=(-1, 1))
    y_scaler_neu = MinMaxScaler(feature_range=(-1, 1))

    Y_ann_s = y_scaler_ann.fit_transform(Ya_tr).astype("float32")
    Y_neu_s = y_scaler_neu.fit_transform(Yn_tr).astype("float32")
    plot_targets(Y_ann_s, Y_neu_s, "minmaxscaler")
 

    print("Y_ann_s:", Y_ann_s)
    print("len(Y_ann_s):", len(Y_ann_s))
    print("Y_neu_s:", Y_neu_s)
    print("X_s:", X_s)
    print("len(X_s):", len(X_s))
    #train annihilation model

    #X_s = tf.stack(X_s)
    #Y_ann_s = tf.stack(Y_ann_s)
    #Y_neu_s = tf.stack(Y_neu_s)
    print("dim:", X_s.ndim)
    print("dim2:", Y_ann_s.ndim)
    print("shape:", X_s.shape)
    print("shape2:", Y_ann_s.shape)
    #X_tr, X_val, Ya_tr, Ya_val = train_test_split(X_s, Y_ann_s, test_size=0.2, random_state=150)
    model_ann = build_regressor(out_dim=4)
    ckpt_ann = ModelCheckpoint("weights_ann.weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    #model_ann.fit(X_tr, Ya_tr, validation_data=(X_val, Ya_val), epochs=100, batch_size=128, callbacks=[ckpt_ann, es], verbose=2)
    model_ann.fit(X_s, Y_ann_s, validation_data=(X_val, Ya_val), epochs=100, batch_size=128, callbacks=[ckpt_ann, es], verbose=2)
    #train neutroncapture model
    #X_tr, X_val, Yn_tr, Yn_val = train_test_split(X_s, Y_neu_s, test_size=0.2, random_state=150)
    model_neu = build_regressor(out_dim=4)
    ckpt_neu = ModelCheckpoint("weights_neu.weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
    #model_neu.fit(X_tr, Yn_tr, validation_data=(X_val, Yn_val), epochs=100, batch_size=128, callbacks=[ckpt_neu, es], verbose=2)
    model_neu.fit(X_s, Y_neu_s, validation_data=(X_val, Yn_val), epochs=100, batch_size=128, callbacks=[ckpt_neu, es], verbose=2)

    dump(sc_x, "scaler_x.joblib")
    dump(sc_y, "scaler_y.joblib")
    dump(sc_z, "scaler_z.joblib")
    dump(sc_t, "scaler_t.joblib")
    dump(y_scaler_ann, "y_scaler_ann.joblib")
    dump(y_scaler_neu, "y_scaler_neu.joblib")

    print("Saved: weights_ann.h5, weights_neu.h5, input scalers, target scalers (ann/neu).")

if __name__ == "__main__":
    main()
