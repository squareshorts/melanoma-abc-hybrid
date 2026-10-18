import xgboost as xgb

def train_xgb(X, y, cfg_xgb: dict):
    model = xgb.XGBClassifier(
        n_estimators=int(cfg_xgb["n_estimators"]),
        max_depth=int(cfg_xgb["max_depth"]),
        learning_rate=float(cfg_xgb["learning_rate"]),
        subsample=float(cfg_xgb.get("subsample", 1.0)),
        colsample_bytree=float(cfg_xgb.get("colsample_bytree", 1.0)),
        reg_lambda=float(cfg_xgb.get("reg_lambda", 1.0)),
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def predict_proba(model, X):
    return model.predict_proba(X)[:,1]
