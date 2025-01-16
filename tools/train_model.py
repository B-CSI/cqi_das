import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import joblib


# 1. Load feature datasets
candas2_feat = pd.read_csv("features2/candas2_features.csv", index_col=0)
safe_t_feat = pd.read_csv("features2/safe_t_features.csv", index_col=0)
castor2_feat = pd.read_csv("features2/castor2_features.csv", index_col=0)
twin_gc_feat = pd.read_csv("features2/twin_gc_features.csv", index_col=0)
twin_tf_feat = pd.read_csv("features2/twin_tf_features.csv", index_col=0)
tf_2nd_pick_feat = pd.read_csv("features2/tf_2nd_pick_features.csv", index_col=0)
tf_same_day_feat = pd.read_csv("features2/tf_same_day_features.csv", index_col=0)
castor_feat = pd.read_csv("features2/castor_features.csv", index_col=0)
candas1_feat = pd.read_csv("features2/candas1_features.csv", index_col=0)
safe2_feat = pd.read_csv("features2/safe2_features.csv", index_col=0)

# 2. Create a combined training dataset
data_train = pd.concat(
    [
        castor2_feat,
        twin_gc_feat,
        twin_tf_feat,
        tf_2nd_pick_feat,
        tf_same_day_feat,
        castor_feat,
        candas1_feat,
        candas2_feat,
        safe_t_feat,
        safe2_feat,
    ],
    axis=0,
    ignore_index=True,
)

# fmt: off
selected_features = [
    "env-impulse-factor",           # 1
    "env-margin-factor",            # 1
    "env-peak",                     # 1
    "env_freq-avg",                 # 1
    "psd-kurtosis",                 # 1
    "env-variance",                 # 6
    "1-peak-prominence-factor",     # 7
    "env-psd-entropy",              # 8
    "psd-entropy",                  # 9
    "3-peak-prominence-factor",     # 10
    "psd-skew",                     # 11
    "env-median",                   # 12
    "env-clearance-factor",         # 13
    "env-rms",                      # 14
    "freq-skew",                    # 15
    "mfcc1_mean",                   # 16
    "mfcc1_max",                    # 17
]
# fmt: on

# 3. Separate features and target
features = data_train.drop(columns="target")[selected_features]
target = data_train["target"]

# 4. IQR-based outlier removal
mask = pd.Series(True, index=features.index)

for col in features.columns:
    Q1 = features[col].quantile(0.05)
    Q3 = features[col].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    col_mask = features[col].between(lower_bound, upper_bound, inclusive="both")
    mask = mask & col_mask  # retain rows that are in-bounds for all columns

clean_features = features[mask]
clean_target = target[mask]
data_cleaned = pd.concat([clean_features, clean_target], axis=1)

# 5. Train/test split
X = data_cleaned.drop(columns="target")
y = data_cleaned["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# 6. Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

X_train_subset = X_train_scaled
X_test_subset = X_test_scaled

# 8. Define and train XGBoost model
xgb_clf_best = xgb.XGBClassifier(
    learning_rate=0.05,
    max_depth=4,
    gamma=5,
    n_estimators=500,
    subsample=0.75,
    colsample_bytree=0.5,
    min_child_weight=3,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="auc",
    reg_lambda=10,
)

xgb_clf_best.fit(X_train_subset, y_train, eval_set=[(X_test_subset, y_test)], verbose=False)

# 9. Calibrate model
calibrated_clf = CalibratedClassifierCV(estimator=xgb_clf_best, method="isotonic", cv=5)
calibrated_clf.fit(X_train_subset, y_train, eval_set=[(X_test_subset, y_test)], verbose=False)

# 10. Save model and scaler
joblib.dump(calibrated_clf, "calibrated_xgb.pkl")
joblib.dump(scaler, "robust_scaler.pkl")
