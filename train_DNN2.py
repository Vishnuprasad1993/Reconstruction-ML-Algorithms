
# Usage:
#   python train_DNN2.py sipmHits_output.csv annihilationEvents_output.csv
#   python train_DNN2.py sipmHits_output.csv neutronCaptureEvents_output.csv


import sys
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Masking, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

MAX_HITS            = 600      
SPAN_NS             = 100.0    
CONSECUTIVE_GAP_NS  = 2000.0   
DETECT_RESETS       = True     
RESET_TOL_NS        = 1e-6    
MIN_HITS_PER_EVENT  = 3       
SORT_GLOBALLY       = False    
SEED                = 150
np.random.seed(SEED)

def _parse_time_cell_to_float(x) -> float:
    s = str(x).strip().replace(',', ' ')
    tok = s.split()
    return float(tok[0]) if tok else np.nan


def load_hits_raw(hits_csv: str) -> pd.DataFrame:
    
    df = pd.read_csv(hits_csv)

    col_map = {
        "Sipm_Hit_XPosition": "X(mm)",
        "Sipm_Hit_YPosition": "Y(mm)",
        "Sipm_Hit_ZPosition": "Z(mm)",
    }
    for k, v in col_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    if "SipmTime(ns)" in df.columns:
        df["Time(ns)"] = df["SipmTime(ns)"].apply(_parse_time_cell_to_float).astype("float32")
    elif "Time(ns)" in df.columns:
        df["Time(ns)"] = df["Time(ns)"].apply(_parse_time_cell_to_float).astype("float32")
    else:
        raise ValueError("hits CSV must contain 'SipmTime(ns)' or 'Time(ns)'")

    need = {"X(mm)","Y(mm)","Z(mm)","Time(ns)"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"hits CSV missing columns: {missing}")

    return df[["X(mm)","Y(mm)","Z(mm)","Time(ns)"]]


def load_labels_one(labels_csv: str) -> pd.DataFrame:
    
    lab = pd.read_csv(labels_csv, usecols=["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]).reset_index(drop=True)
    lab = lab.rename(columns={"X(mm)":"tx","Y(mm)":"ty","Z(mm)":"tz","Time(ns)":"tt"})
    lab["event_id"] = np.arange(len(lab), dtype=np.int64)
    return lab[["event_id","tx","ty","tz","tt"]]

def assign_event_ids_by_span(
    hits_df: pd.DataFrame,
    span_ns: float = SPAN_NS,
    detect_resets: bool = DETECT_RESETS,
    reset_tol_ns: float = RESET_TOL_NS,
    consecutive_gap_ns: float | None = CONSECUTIVE_GAP_NS,
    sort_globally: bool = SORT_GLOBALLY,
) -> pd.DataFrame:
    
    df = hits_df.copy()

    if sort_globally:
        df = df.sort_values("Time(ns)", kind="mergesort").reset_index(drop=True)

    t = df["Time(ns)"].astype(float).to_numpy()
    n = len(df)
    if n == 0:
        df["event_id"] = []
        return df

    eids = np.empty(n, dtype=np.int64)
    eid = 0
    start = 0
    eids[0] = eid

    for i in range(1, n):
        dt = t[i] - t[i-1]
        reset = detect_resets and (dt < -reset_tol_ns)
        big_gap = (consecutive_gap_ns is not None) and (dt > consecutive_gap_ns)
        span = t[i] - t[start]

        if reset or big_gap or (span > span_ns):
            eid += 1
            start = i
        eids[i] = eid

    df["event_id"] = eids
    return df


def filter_min_hits(df_with_eid: pd.DataFrame, min_hits: int = MIN_HITS_PER_EVENT) -> pd.DataFrame:
    if min_hits <= 1:
        unique = sorted(df_with_eid["event_id"].unique())
        remap = {old:i for i, old in enumerate(unique)}
        out = df_with_eid.copy()
        out["event_id"] = out["event_id"].map(remap)
        return out

    sizes = df_with_eid.groupby("event_id").size()
    keep = sizes[sizes >= min_hits].index
    out = df_with_eid[df_with_eid["event_id"].isin(keep)].copy()
    if out.empty:
        return out
    unique = sorted(out["event_id"].unique())
    remap = {old:i for i, old in enumerate(unique)}
    out["event_id"] = out["event_id"].map(remap)
    return out

def build_event_tensor_from_hits(hits_eid: pd.DataFrame, max_hits: int = MAX_HITS):
    
    X_list, eids = [], []
    for eid, g in hits_eid.groupby("event_id", sort=True):
        arr = g[["X(mm)","Y(mm)","Z(mm)","Time(ns)"]].to_numpy(np.float32).T  # (4, n_hits)
        if arr.shape[1] > max_hits:
            arr = arr[:, :max_hits]
        if arr.shape[1] < max_hits:
            arr = np.pad(arr, ((0,0),(0, max_hits - arr.shape[1])), mode="constant", constant_values=np.nan)
        X_list.append(arr)
        eids.append(eid)
    if not X_list:
        raise ValueError("No events to build. Relax MIN_HITS_PER_EVENT or increase SPAN_NS/CONSECUTIVE_GAP_NS.")
    X = np.stack(X_list, axis=0)  
    return np.asarray(eids, dtype=np.int64), X


def build_regressor(out_dim: int = 4, input_shape=(4, MAX_HITS)) -> keras.Model:
    keras.backend.clear_session()
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        Flatten(),
        Dense(419, kernel_initializer='he_normal', activation='relu'),
        Dropout(0.43),
        Dense(369, kernel_initializer='he_normal', activation='relu'),
        Dropout(0.07),
        Dense(150, kernel_initializer='he_normal', activation='linear'),
        Dropout(0.50),
        Dense(out_dim, kernel_initializer='he_uniform', activation='linear'),
    ])
    model.compile(optimizer=keras.optimizers.Adamax(2e-3), loss="mse", metrics=["mae"])
    return model

def main():
    if len(sys.argv) < 3:
        print("Usage: python train_DNN2.py sipmHits_output.csv labels.csv")
        sys.exit(1)

    hits_csv, labels_csv = sys.argv[1], sys.argv[2]

    hits_raw = load_hits_raw(hits_csv)          
    lab = load_labels_one(labels_csv)           

    hits_eid = assign_event_ids_by_span(
        hits_raw,
        span_ns=SPAN_NS,
        detect_resets=DETECT_RESETS,
        reset_tol_ns=RESET_TOL_NS,
        consecutive_gap_ns=CONSECUTIVE_GAP_NS,
        sort_globally=SORT_GLOBALLY,
    )

    sizes_all = hits_eid.groupby("event_id").size()
    print("Clustering (pre-filter): events =", sizes_all.shape[0],
          "mean hits =", float(sizes_all.mean()),
          "median =", float(sizes_all.median()),
          "p10 =", float(sizes_all.quantile(0.10)),
          "p90 =", float(sizes_all.quantile(0.90)))

    hits_eid = filter_min_hits(hits_eid, min_hits=MIN_HITS_PER_EVENT)
    if hits_eid.empty:
        raise ValueError(
            "All clusters were filtered out. Lower MIN_HITS_PER_EVENT or increase SPAN_NS/CONSECUTIVE_GAP_NS."
        )

    sizes = hits_eid.groupby("event_id").size()
    print("After filtering: events =", sizes.shape[0],
          "| mean hits =", float(sizes.mean()),
          "| median =", float(sizes.median()),
          "| p10 =", float(sizes.quantile(0.10)),
          "| p90 =", float(sizes.quantile(0.90)))

    event_ids, X_all = build_event_tensor_from_hits(hits_eid, MAX_HITS)
    X_all = np.nan_to_num(X_all, nan=0.0)

    
    n_events_hits = len(event_ids)
    n_events_lab  = len(lab)
    N = min(n_events_hits, n_events_lab)
    if n_events_hits != n_events_lab:
        print(f"[Info] Event count mismatch: hits={n_events_hits}, labels={n_events_lab}. "
              f"Truncating to {N} (assuming same order).")

    order = np.argsort(event_ids)
    event_ids = event_ids[order][:N]
    X_all = X_all[order][:N]
    labN = lab.iloc[:N].reset_index(drop=True)
    Y_all = labN[["tx","ty","tz","tt"]].to_numpy(np.float32)

    print("Tensor shapes : X_all:", X_all.shape, "| Y_all:", Y_all.shape)

    X_tr, X_val, Y_tr, Y_val = train_test_split(X_all, Y_all, test_size=0.2, random_state=SEED)

    sc_vtx = MinMaxScaler(feature_range=(-1, 1))
    sc_t   = MinMaxScaler(feature_range=(-1, 1))

    vtx_tr  = X_tr[:, 0:3, :]  
    t_tr    = X_tr[:, 3, :]    
    vtx_val = X_val[:, 0:3, :]
    t_val   = X_val[:, 3, :]

    vtx_tr_s  = sc_vtx.fit_transform(vtx_tr.reshape(-1, vtx_tr.shape[-1])).reshape(vtx_tr.shape)
    t_tr_s    = sc_t.fit_transform(t_tr)
    vtx_val_s = sc_vtx.transform(vtx_val.reshape(-1, vtx_val.shape[-1])).reshape(vtx_val.shape)
    t_val_s   = sc_t.transform(t_val)

    def stack_feats(vtx, t):
        return np.stack([vtx[:, 0, :], vtx[:, 1, :], vtx[:, 2, :], t], axis=1)

    X_tr_s  = stack_feats(vtx_tr_s,  t_tr_s)
    X_val_s = stack_feats(vtx_val_s, t_val_s)

    Y_tr = Y_tr.copy(); Y_val = Y_val.copy()
    Y_tr[:, 3]  = np.log1p(np.clip(Y_tr[:, 3], 0, None))
    Y_val[:, 3] = np.log1p(np.clip(Y_val[:, 3], 0, None))

    sc_y = MinMaxScaler(feature_range=(-1, 1))
    Y_tr_s = sc_y.fit_transform(Y_tr).astype("float32")
    Y_val_s = sc_y.transform(Y_val).astype("float32")

    model = build_regressor(out_dim=4, input_shape=(4, MAX_HITS))
    ckpt = ModelCheckpoint("weights_best.weights.h5", monitor="val_loss",
                           save_best_only=True, save_weights_only=True, verbose=1)
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    history = model.fit(
        X_tr_s, Y_tr_s,
        validation_data=(X_val_s, Y_val_s),
        epochs=100, batch_size=128,
        callbacks=[ckpt, es],
        verbose=2
    )

    dump(sc_vtx, "x_scaler_vtx.joblib")
    dump(sc_t,   "x_scaler_t.joblib")
    dump(sc_y,   "y_scaler.joblib")
    print("Saved: weights_best.weights.h5, x_scaler_vtx.joblib, x_scaler_t.joblib, y_scaler.joblib.")


if __name__ == "__main__":
    main()
