# ---------------- USER CONFIG ----------------
energy = '9631899'
species = ['o']
release = 'rel05'
number_history = 2
raw_feature_names = ['symh','asyh','asyd','ae','f10.7','kp','swp','swn','swv','by','bz']
stacked_model = 'pipeline_results'
test_ts = '2017-01-01'
test_te = '2018-01-01'
os.makedirs(stacked_model, exist_ok=True)

# L groups and sectoring
l_sets = [(1,3),(4,6),(7,8)]
# XGBoost / LightGBM params (starting sensible defaults)
XGB_PARAMS = {
    'objective':'reg:squarederror',
    'eta':0.05,
    'max_depth':8,
    'subsample':0.9,
    'colsample_bytree':0.8,
    'reg_lambda':2.0,
    'seed':42,
    'eval_metric':'rmse'
}
LGB_PARAMS = {
    'objective':'regression',
    'learning_rate':0.05,
    'num_leaves':90,
    'feature_fraction':0.8,
    'bagging_fraction':0.8,
    'bagging_freq':1,
    'lambda_l2':2.0,
    'metric':'rmse',
    'verbosity':-1
}
# early stopping rounds
XGB_EARLY_STOP = 80

print("=== START fixed pipeline ===")
# 1) Load data via your helper pipeline
fulldata_directories, fulldataset_csv, fulldata_settings = initialize_fulldata_var(release=release,
                                                                                  average_time=300,
                                                                                  raw_coor_names=["mlt","l","lat"],
                                                                                  coor_names=["cos0",'sin0','scaled_lat','scaled_l'],
                                                                                  raw_feature_names=raw_feature_names,
                                                                                  number_history=number_history,
                                                                                  history_resolution=2*3600.,
                                                                                  energy=[energy],
                                                                                  species=species)
df_full = read_probes_data(fulldata_directories["rawdata_dir"], fulldata_settings)
df_coor, fulldata_settings = scale_corrdinates(df_full, fulldata_settings, fulldata_settings["datetime_name"], fulldataset_csv["df_coor"], save_data=False, plot_data=False)
df_y, df_full, fulldata_settings = load_y(fulldata_directories, fulldataset_csv, fulldata_settings, recalc=False, df_full=df_full, save_data=False, plot_data=False, energy_bins=[energy], species_arr=species)
df_features_history, df_full, fulldata_settings = load_features(fulldata_directories, fulldataset_csv, fulldata_settings, recalc=False, df_full=df_full, save_data=False, plot_data=False, raw_feature_names=raw_feature_names)

# fig, axes = plt.subplots(1, 4, figsize=(12, 5))

# assemble
df_data = pd.concat([df_y.reset_index(drop=True),
                     df_coor[fulldata_settings['coor_names']].reset_index(drop=True),
                     df_features_history.reset_index(drop=True)], axis=1)

# sns.histplot(df_data['log_o_flux_9631899'], bins=50, kde=True, ax=axes[0], color="skyblue")
# axes[0].set_title("Histogram of O+ Flux (Raw)")
# axes[0].set_xlabel("O+ Flux")
# axes[0].set_ylabel("Count")

# sns.histplot(df_data['log_o_flux_9631899'], bins=50, kde=True, ax=axes[0], color="skyblue")
# axes[1].set_title("Histogram of O+ Flux (Raw)")
# axes[1].set_xlabel("O+ Flux")
# axes[1].set_ylabel("Count")

# attach raw columns if available (Vsw, bz, etc.)
for c in ['l','mlt','bz','swv','swn','swp','kp','ae','f10.7','symh']:
    if c in df_full.columns:
        df_data[c] = df_full[c].values

# sanity checks
log_y_name = f'log_{species[0]}_flux_{energy}'
for r in [log_y_name, 'l', 'mlt']:
    if r not in df_data.columns:
        raise RuntimeError(f"Required column {r} missing in assembled df_data")

# keep only rows with essential fields
df_data = df_data.dropna(subset=[log_y_name, 'l', 'mlt'])
df_data['L_floor'] = np.floor(df_data['l']).astype(int)
df_data['mlt_sector'] = df_data['mlt'].apply(assign_sector)

mask = df_data["log_o_flux_9631899"] == 1
df_data = df_data[~mask]

# 2) Derived physics features (keep only those in the validated minimal set)
# Bz_neg, E_conv_0h, P_dyn, Vsw proxy
if 'bz' in df_data.columns:
    df_data['Bz_neg_0h'] = np.maximum(-df_data['bz'].astype(float), 0.0)
elif 'scaled_bz_0h' in df_data.columns:
    df_data['Bz_neg_0h'] = np.maximum(-df_data['scaled_bz_0h'].astype(float), 0.0)
if 'swv' in df_data.columns:
    df_data['Vsw'] = df_data['swv'].astype(float)
if 'swn' in df_data.columns:
    df_data['Nsw'] = df_data['swn'].astype(float)
if 'Vsw' in df_data.columns and 'Bz_neg_0h' in df_data.columns:
    df_data['E_conv_0h'] = df_data['Vsw'] * df_data['Bz_neg_0h']
if 'Nsw' in df_data.columns and 'Vsw' in df_data.columns:
    df_data['P_dyn'] = 1.6726e-6 * df_data['Nsw'] * (df_data['Vsw']**2)

# print("1"+ df_data.columns.tolist())


# 3) Feature list
candidate_features = [
    'scaled_kp_0h', 'scaled_ae_0h', 'scaled_symh_0h', 'scaled_f10.7_0h',
    'Vsw', 'scaled_bz_0h', 'Bz_neg_0h', 'E_conv_0h', 'P_dyn',
    'scaled_l', 'scaled_lat', 'cos0', 'sin0'
]

# only keep existing columns (some names may be slightly different; adapt if needed)
features = [f for f in candidate_features if f in df_data.columns]
print("Using pruned feature set (count):", len(features))
print(features)

# sns.histplot(df_data['log_o_flux_9631899'], bins=50, kde=True, ax=axes[0], color="skyblue")
# axes[2].set_title("Histogram of O+ Flux (Raw)")
# axes[2].set_xlabel("O+ Flux")
# axes[2].set_ylabel("Count")

# 4) drop rows lacking pruned features or target
df_data = df_data.dropna(subset=[log_y_name] + features).reset_index(drop=True)
print("Rows remaining:", len(df_data))
# print("2"+ df_data.columns.tolist())

# sns.histplot(df_data['log_o_flux_9631899'], bins=50, kde=True, ax=axes[0], color="skyblue")
# axes[3].set_title("Histogram of O+ Flux (Raw)")
# axes[3].set_xlabel("O+ Flux")
# axes[3].set_ylabel("Count")
# print("3"+ df_data.columns.tolist())

# plt.tight_layout()
# plt.show()

# 6) splitting data
df_data[fulldata_settings['datetime_name']] = pd.to_datetime(df_data[fulldata_settings['datetime_name']])

mask_test = (df_data[fulldata_settings['datetime_name']] >= pd.to_datetime(test_ts)) & \
            (df_data[fulldata_settings['datetime_name']] <= pd.to_datetime(test_te))
idx_test = df_data.index[mask_test].to_numpy()
idx_trainval = df_data.index[~mask_test].to_numpy()

print("Rows reserved for TEST (time-based):", len(idx_test))
print("Rows available for train+val:", len(idx_trainval))

# Build train/val split stratified by L_floor on the remaining data
# We call train_test_split on the list of absolute indices so idx_train/idx_val are global indices.
# Build train/val split stratified by L_floor
try:
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=0.3,
        random_state=42,
        stratify=df_data.loc[idx_trainval, 'L_floor'].values
    )
except Exception as e:
    print("Stratified train/val on trainval failed, falling back to random split:", e)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.3, random_state=42)

df_data["y_clip"] = df_data["log_o_flux_9631899"]
# Use raw target values (no clipping)
log_y_name = f'log_{species[0]}_flux_{energy}'

X_train = df_data.loc[idx_train, features].astype(float).values
y_train = df_data.loc[idx_train, log_y_name].astype(float).values

X_val   = df_data.loc[idx_val,   features].astype(float).values
y_val   = df_data.loc[idx_val,   log_y_name].astype(float).values

X_test  = df_data.loc[idx_test,  features].astype(float).values
y_test  = df_data.loc[idx_test,  log_y_name].astype(float).values

# -------------------------------------------------------------------------------
print("Sizes train/val/test:", len(y_train), len(y_val), len(y_test))

# Build datasets for XGBoost / LightGBM
dtrain_xgb = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval_xgb   = xgb.DMatrix(X_val,   label=y_val,   feature_names=features)
dtrain_lgb = lgb.Dataset(X_train, label=y_train)
dval_lgb   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain_lgb)

# Also prepare a test DMatrix for later evaluation
# dtest_xgb = xgb.DMatrix(X_test, feature_names=features)
# keep idx arrays for later mapping back to df_data
# idx_train, idx_val, idx_test are numpy arrays of absolute df_data indices
# --------------------------------------------------------------
# 7) Train global models with the tuned params (validate on val)
print("Training global XGBoost...")
bst_xgb = xgb.train(XGB_PARAMS, dtrain_xgb, num_boost_round=10000, evals=[(dval_xgb,'val')], early_stopping_rounds=XGB_EARLY_STOP, verbose_eval=100)
joblib.dump(bst_xgb, os.path.join(stacked_model,'global_xgb_fixed.pkl'))

print("Training global LightGBM...")
early_stopping_cb = lgb.early_stopping(stopping_rounds=80)
bst_lgb = lgb.train(LGB_PARAMS, dtrain_lgb, num_boost_round=10000, valid_sets=[dval_lgb], callbacks=[early_stopping_cb, lgb.log_evaluation(100)])
joblib.dump(bst_lgb, os.path.join(stacked_model,'global_lgb_fixed.pkl'))

# Predictions on validation only
y_val_xgb = bst_xgb.predict(dval_xgb)
y_val_lgb = bst_lgb.predict(X_val)

# quick correlation heatmaps (validation)
plot_correlation_heatmap(y_val, y_val_xgb,
                        xrange=[0,9],
                        figname=os.path.join(stacked_model,'val_corr_xgb'),
                        data_type='XGB val')

residuals = y_val - y_val_xgb

plt.figure(figsize=(10,6))
plt.scatter(y_val, residuals, alpha=0.5, color="blue")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals (True - Predicted)")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()

plot_correlation_heatmap(y_val, y_val_lgb,
                        xrange=[0,9],
                        figname=os.path.join(stacked_model,'val_corr_lgb'),
                        data_type='LGB val')

residuals = y_val - y_val_lgb

plt.figure(figsize=(10,6))
plt.scatter(y_val, residuals, alpha=0.5, color="blue")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals (True - Predicted)")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()

print("Global XGB val R2:", r2_score(y_val, y_val_xgb))
print("Global LGB val R2:", r2_score(y_val, y_val_lgb))

# 8) Train per-L group models and for L=4-6 per-sector models (validate only)
perL_models = {}
perL_metrics = []
for (lmin, lmax) in l_sets:
    # group mask across all rows (train+val+test) for counting/selection
    mask = (df_data['L_floor'] >= lmin) & (df_data['L_floor'] <= lmax)
    n_total = int(mask.sum())

    # restrict to TRAIN rows only for actual training
    train_mask_grp = mask & df_data.index.isin(idx_train)
    n_train = int(train_mask_grp.sum())
    print(f"Training L group {lmin}-{lmax} on TRAIN only (n_train={n_train}, total_in_group={n_total})")

    # training data (TRAIN only)
    df_grp_train = df_data.loc[train_mask_grp]
    Xg_train = df_grp_train[features].values
    yg_train = df_grp_train['y_clip'].values

    # Use VAL rows *within the group* as validation if available (preferred)
    val_mask_grp = mask & df_data.index.isin(idx_val)
    n_val = int(val_mask_grp.sum())

    if n_val >= 50:
        # Use group-specific val rows from the global val set
        Xg_val = df_data.loc[val_mask_grp, features].values
        yg_val = df_data.loc[val_mask_grp, 'y_clip'].values
    else:
        # Fallback: small internal split from TRAIN only (still safe — test rows are not used)
        Xg_train, Xg_val, yg_train, yg_val = train_test_split(Xg_train, yg_train, test_size=0.2, random_state=42)

    dgtrain = xgb.DMatrix(Xg_train, label=yg_train, feature_names=features)
    dgval   = xgb.DMatrix(Xg_val,   label=yg_val,   feature_names=features)
    bst_grp = xgb.train(XGB_PARAMS, dgtrain, num_boost_round=8000, evals=[(dgval,'val')], early_stopping_rounds=50, verbose_eval=False)

    perL_models[(lmin,lmax)] = {'xgb': bst_grp, 'sectors': {}}
    r2_v = r2_score(yg_val, bst_grp.predict(dgval))
    perL_metrics.append({'group':f'{lmin}-{lmax}','n_train':int(n_train),'n_total':int(n_total),'val_r2':float(r2_v)})
    print(f"  group val_r2={r2_v:.3f} (n_train={n_train}, n_val={n_val})")

    # Special-case: build per-sector models for L=4-6 — train on TRAIN-only rows within the sector
    if (lmin,lmax) == (4,6):
        # get TRAIN rows for this L and compute sectors on them
        df_grp_train = df_grp_train.copy()
        df_grp_train['mlt_sector'] = df_grp_train['mlt'].apply(assign_sector)

        for sector in ['dawn','day','dusk','night']:
            mask_s_train = df_grp_train['mlt_sector'] == sector
            ns_train = int(mask_s_train.sum())
            if ns_train < 500:
                print(f"   skip sector {sector} (train n={ns_train})")
                continue

            # sector training data (TRAIN only)
            Xs_train = df_grp_train.loc[mask_s_train, features].values
            ys_train = df_grp_train.loc[mask_s_train, 'y_clip'].values

            # prefer VAL-sector rows from global VAL if available
            val_mask_s = (mask & df_data.index.isin(idx_val)) & (df_data['mlt'].apply(assign_sector) == sector)
            n_val_s = int(val_mask_s.sum())

            if n_val_s >= 50:
                Xs_val = df_data.loc[val_mask_s, features].values
                ys_val = df_data.loc[val_mask_s, 'y_clip'].values
            else:
                Xs_train, Xs_val, ys_train, ys_val = train_test_split(Xs_train, ys_train, test_size=0.2, random_state=42)

            ds_train = xgb.DMatrix(Xs_train, label=ys_train, feature_names=features)
            ds_val   = xgb.DMatrix(Xs_val,   label=ys_val,   feature_names=features)
            bst_sec = xgb.train(XGB_PARAMS, ds_train, num_boost_round=6000, evals=[(ds_val,'val')], early_stopping_rounds=40, verbose_eval=False)
            perL_models[(lmin,lmax)]['sectors'][sector] = {'model': bst_sec, 'n_train': int(ns_train)}
            print(f"    sector {sector} val_r2={r2_score(ys_val, bst_sec.predict(ds_val)):.3f} n_train={ns_train}")


pd.DataFrame(perL_metrics).to_csv(os.path.join(stacked_model,'perL_metrics_fixed.csv'), index=False)

# 9) Build per-L predictions for VAL using safe mapping
perL_val = compute_perL_preds_for_indices(idx_val, df_data, perL_models, features, bst_xgb)


# --- per-L linear calibration: add after y_val_xgb, y_val_lgb, perL_val computed ---

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

# usage (after val_df built)
df_val_local = df_data.loc[idx_val].reset_index(drop=True)   # ensure alignment
# fit calibrators for each base model
cal_xgb = fit_perL_calibrators(df_val_local, y_val, y_val_xgb)
cal_lgb = fit_perL_calibrators(df_val_local, y_val, y_val_lgb)
cal_perL = fit_perL_calibrators(df_val_local, y_val, perL_val)

# apply calibrations
y_val_xgb_cal  = apply_calibration(y_val_xgb, df_val_local, cal_xgb)
y_val_lgb_cal  = apply_calibration(y_val_lgb, df_val_local, cal_lgb)
perL_val_cal   = apply_calibration(perL_val,  df_val_local, cal_perL)

# Replace usage: use calibrated preds for stacking and summary
val_df = pd.DataFrame({
    'y_true': y_val,
    'xgb_global': y_val_xgb_cal,
    'lgb_global': y_val_lgb_cal,
    'perL_pred': perL_val_cal
})

calibrators_bundle = {'xgb': cal_xgb, 'lgb': cal_lgb, 'perL': cal_perL}

# coarse grid weights (optional) — optimize on validation
best_grid = {'w':None,'r2': -1e9}
ws = np.linspace(0,1,21)
for w1 in ws:
    for w2 in ws:
        if w1 + w2 > 1.0: continue
        w3 = 1.0 - w1 - w2
        pred = w1*val_df['xgb_global'].values + w2*val_df['lgb_global'].values + w3*val_df['perL_pred'].values
        r2v = r2_score(val_df['y_true'].values, pred)
        if r2v > best_grid['r2']:
            best_grid = {'w':(w1,w2,w3),'r2':r2v}
print("Best grid ensemble weights (val):", best_grid['w'], "val R2:", best_grid['r2'])

# Ridge stacking meta-model trained on validation meta-features
X_meta_val = val_df[['xgb_global','lgb_global','perL_pred']].values
y_meta_val = val_df['y_true'].values
ridge = RidgeCV(alphas=np.logspace(-6,2,30), cv=5)
ridge.fit(X_meta_val, y_meta_val)
meta_w = ridge.coef_
intercept = ridge.intercept_
print("Ridge stacking weights (val):", meta_w, "intercept:", intercept)

# Evaluate ensembles on VAL (no test evaluations)
yval_grid  = best_grid['w'][0]*val_df['xgb_global'].values + best_grid['w'][1]*val_df['lgb_global'].values + best_grid['w'][2]*val_df['perL_pred'].values
yval_stack = ridge.predict(X_meta_val)
print("Grid ensemble val R2:", r2_score(val_df['y_true'].values, yval_grid), "MSE:", mean_squared_error(val_df['y_true'].values, yval_grid))
print("Stacked (Ridge) val R2:", r2_score(val_df['y_true'].values, yval_stack), "MSE:", mean_squared_error(val_df['y_true'].values, yval_stack))

plot_correlation_heatmap(val_df['y_true'].values, yval_grid,
                     xrange=[0,9],
                     figname=os.path.join(stacked_model,'val_corr_ensemble_grid'),
                     data_type='ensemble_grid_val')

residuals = val_df['y_true'].values - yval_grid

plt.figure(figsize=(10,6))
plt.scatter(yval_grid, residuals, alpha=0.5, color="blue")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals (True - Predicted)")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()

plot_correlation_heatmap(val_df['y_true'].values, yval_stack,
                     xrange=[0,9],
                     figname=os.path.join(stacked_model,'val_corr_ensemble_stack'),
                     data_type='stack_val')

residuals = val_df['y_true'].values - yval_stack

plt.figure(figsize=(10,6))
plt.scatter(yval_stack, residuals, alpha=0.5, color="blue")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals (True - Predicted)")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.show()

# Save predictions and artifacts for VALIDATION set (instead of test)
val_out = df_data.loc[idx_val].copy().reset_index(drop=True)
val_out['y_true'] = y_val
val_out['xgb_global'] = val_df['xgb_global'].values
val_out['lgb_global'] = val_df['lgb_global'].values
val_out['perL_pred'] = val_df['perL_pred'].values
val_out['ensemble_grid'] = yval_grid
val_out['ensemble_stack'] = yval_stack
val_out.to_csv(os.path.join(stacked_model,'val_predictions_fixed.csv'), index=False)

joblib.dump({'bst_xgb':bst_xgb,'bst_lgb':bst_lgb,'perL_models':perL_models,'ridge_stack':ridge,'features':features, 'calibrators':calibrators_bundle}, os.path.join(stacked_model,'models_fixed.pkl'))

# per-L R2 for stacked predictions on VAL
r2_by_L = val_out.groupby('L_floor').apply(lambda g: r2_score(g['y_true'], g['ensemble_stack']) if len(g)>10 else np.nan)
r2_by_L.to_csv(os.path.join(stacked_model,'r2_by_L_stacked_fixed_val.csv'))
print("Per-L stacked R2 (saved, val):")
print(r2_by_L)

# diagnostic heatmap using VAL
val_out['mlt_hour'] = np.floor(val_out['mlt']).astype(int)
val_out['error'] = val_out['y_true'] - val_out['ensemble_stack']
heat = val_out.pivot_table(index='L_floor', columns='mlt_hour', values='error', aggfunc='mean')

plt.figure(figsize=(12,6))
plt.imshow(heat.fillna(0), aspect='auto', cmap='coolwarm', vmin=-np.nanmax(np.abs(heat.values)), vmax=np.nanmax(np.abs(heat.values)))
plt.colorbar(label='residual mean (obs - pred)')
plt.xlabel('MLT hour'); plt.ylabel('L_floor'); plt.title('Residual mean heatmap (stacked fixed, VAL)')
plt.xticks(np.arange(24))
plt.yticks(np.arange(len(heat.index)), heat.index)
plt.savefig(os.path.join(stacked_model,'resid_heatmap_stacked_fixed_val.png'), dpi=200, bbox_inches='tight')
plt.close()

# summary (validation-only metrics)
summary = {
    'global_xgb_val_r2': float(r2_score(y_val, y_val_xgb_cal)),
    'global_lgb_val_r2': float(r2_score(y_val, y_val_lgb_cal)),
    'grid_val_r2': float(best_grid['r2']),
    'grid_val_mse': float(mean_squared_error(val_df['y_true'].values, yval_grid)),
    'stacked_val_r2': float(r2_score(val_df['y_true'].values, yval_stack)),
    'stacked_val_mse': float(mean_squared_error(val_df['y_true'].values, yval_stack)),
    'n_total': int(len(df_data))
}
pd.DataFrame([summary]).to_csv(os.path.join(stacked_model,'run_summary_fixed_val.csv'), index=False)

print("Saved results in", stacked_model)
print("Total time (s):", time.time() - t0)
print("=== DONE fixed pipeline (VALIDATION ONLY) ===")
