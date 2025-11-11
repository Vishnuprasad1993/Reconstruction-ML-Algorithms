import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(pred_file, true_file, label):
    Y_pred = pd.read_csv(pred_file).values
    Y_true = pd.read_csv(true_file, usecols=["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]).values

    n = min(len(Y_pred), len(Y_true))
    Y_pred, Y_true = Y_pred[:n], Y_true[:n]

    rmse = np.sqrt(mean_squared_error(Y_true, Y_pred, multioutput="raw_values"))
    mae  = mean_absolute_error(Y_true, Y_pred, multioutput="raw_values")
    r2   = r2_score(Y_true, Y_pred, multioutput="raw_values")

    print(f"Evaluation for {label}:")
    print("RMSE [x,y,z,t] :", np.round(rmse, 3))
    print("MAE [x,y,z,t] :", np.round(mae, 3))
    print("R2_Score [x,y,z,t] :", np.round(r2, 3))

def plot_results(pred_file, true_file, label):
    Y_pred = pd.read_csv(pred_file).values
     
    Y_true = pd.read_csv(true_file, usecols=["X(mm)", "Y(mm)", "Z(mm)", "Time(ns)"]).values
    n = min(len(Y_pred), len(Y_true))
    Y_pred = Y_pred[:n]
    Y_true = Y_true[:n]

    mask = np.isfinite(Y_pred).all(axis=1) & np.isfinite(Y_true).all(axis=1)
    Y_pred = Y_pred[mask]
    Y_true = Y_true[mask]

    cols = ["x", "y", "z", "t"]
    
    for i in range(4):
        
        plt.figure(figsize=(6,5))
        plt.scatter(range(len(Y_true[:, i])), Y_true[:, i], 
            color="blue", s=5, alpha=0.6, label="True")
        plt.scatter(range(len(Y_pred[:, i])), Y_pred[:, i], 
            color="orange", s=5, alpha=0.6, label="Predicted")
        #plt.plot([Y_true[:, i].min(), Y_true[:, i].max()],
                 #[Y_true[:, i].min(), Y_true[:, i].max()],
                 #'r--', lw=2)
        plt.xlabel("Event index")
        plt.ylabel(f"{cols[i]}")
        plt.title(f"{label} - {cols[i]} (True vs Predicted)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{label}_{cols[i]}_true_pred_compare.png")
        plt.close()


    errors = Y_pred - Y_true
    for i in range(4):
        plt.figure(figsize=(6,5))
        plt.hist(errors[:, i], bins=100, alpha=0.7)
        plt.xlabel(f"Error in {cols[i]} (Pred - True)")
        plt.ylabel("Counts")
        plt.title(f"{label} - {cols[i]} Error Distribution")
        plt.tight_layout()
        plt.savefig(f"{label}_{cols[i]}_error.png")
        plt.close()



def main():
    if len(sys.argv) < 5:
        print("Usage: python eval_DNN.py predictions_ann.csv annihilationEvents_output.csv predictions_neu.csv neutronCaptureEvents_output.csv")
        sys.exit(1)

    pred_ann, true_ann, pred_neu, true_neu = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    evaluate(pred_ann, true_ann, "Annihilation")
    evaluate(pred_neu, true_neu, "Neutron capture")

    plot_results(pred_ann, true_ann, "Annihilation")
    plot_results(pred_neu, true_neu, "Neutron capture")
    print(f"Saved scatter and error plots")

if __name__ == "__main__":
    main()
