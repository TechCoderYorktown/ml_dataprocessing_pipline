import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb


def assign_sector(mlt):
    # map 0-3 and 21-24 to night
    # {'dawn':(3,9),'day':(9,15),'dusk':(15,21),'night':(21,24)}
    try:
        m = float(mlt)
    except:
        return 'night'
    if 3 <= m < 9: return 'dawn'
    if 9 <= m < 15: return 'day'
    if 15 <= m < 21: return 'dusk'
    return 'night'

def hrs_from_hist_name(c):
    # expects pattern like 'scaled_kp_6h' or 'scaled_symh_0h'
    try:
        return int(c.split('_')[-1].replace('h',''))
    except:
        return 0

def build_aggregates_from_history(df, base_prefixes=['scaled_kp','scaled_ae','scaled_bz','scaled_swv','scaled_symh'], windows=[6,24]):
    for base in base_prefixes:
        cols = [c for c in df.columns if c.startswith(base+'_') and c.endswith('h')]
        if not cols:
            continue
        cols_sorted = sorted(cols, key=hrs_from_hist_name)
        for w in windows:
            sel = [c for c in cols_sorted if hrs_from_hist_name(c) <= w]
            if len(sel) >= 1:
                df[f'{base}_max_{w}h'] = df[sel].max(axis=1)
                df[f'{base}_mean_{w}h'] = df[sel].mean(axis=1)
                df[f'{base}_std_{w}h'] = df[sel].std(axis=1).fillna(0)
                # 0h direct alias if present
                if f'{base}_0h' in cols_sorted:
                    df[f'{base}_0h'] = df[f'{base}_0h']
                # quick change proxy (0h - 2h)
                twoh = [c for c in sel if hrs_from_hist_name(c) == 2]
                if twoh and (f'{base}_0h' in df.columns):
                    df[f'{base}_d02_0h'] = df[f'{base}_0h'] - df[twoh[0]]
    return df

def compute_perL_preds_for_indices(idx_array, df_data, perL_models, features, fallback_model):
    preds = np.zeros(len(idx_array))
    for j, orig_i in enumerate(idx_array):
        Lf = int(df_data.loc[orig_i, 'L_floor'])
        used = False
        for (lmin, lmax), mm in perL_models.items():
            if (Lf >= lmin) and (Lf <= lmax):
                sector = df_data.loc[orig_i,'mlt_sector'] if 'mlt_sector' in df_data.columns else None
                if 'sectors' in mm and sector in mm['sectors']:
                    model = mm['sectors'][sector]['model']
                    xrow = df_data.loc[orig_i, features].values.reshape(1, -1)
                    preds[j] = float(model.predict(xgb.DMatrix(xrow, feature_names=features))[0])
                    used = True
                    break
                model = mm['xgb']
                xrow = df_data.loc[orig_i, features].values.reshape(1, -1)
                preds[j] = float(model.predict(xgb.DMatrix(xrow, feature_names=features))[0])
                used = True
                break
        if not used:
            xrow = df_data.loc[orig_i, features].values.reshape(1, -1)
            preds[j] = float(fallback_model.predict(xgb.DMatrix(xrow, feature_names=features))[0])
    return preds

def safe_savefig(fig, path):
    try:
        fig.savefig(path, dpi=200, bbox_inches='tight')
    except:
        plt.savefig(path, dpi=200, bbox_inches='tight')

def fit_perL_calibrators(df_val, y_true, model_preds, pred_name = ''):
  # df_val: df_data.loc[idx_val].reset_index(drop=True) or aligned series
  # y_true: numpy array
  unique_L = np.sort(df_val['L_floor'].unique())
  calibrators = {}
  for L in unique_L:
    mask = df_val['L_floor'].values == L
    if mask.sum() < 50:   # skip tiny bins
      continue
    X = model_preds[mask].reshape(-1,1)
    y = y_true[mask]
    reg = LinearRegression().fit(X, y)
    calibrators[int(L)] = (reg.coef_[0], reg.intercept_)
  return calibrators

def apply_calibration(preds, df, calibrators):
  out = preds.copy()
  for i in range(len(preds)):
      L = int(df.loc[i, 'L_floor'])
      if L in calibrators:
        a,b = calibrators[L]
        out[i] = a*preds[i] + b
  return out
