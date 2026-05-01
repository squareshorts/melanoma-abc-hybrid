import shap
import xgboost as xgb

def shap_values_tree(model, X):
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        return sv, explainer
    except ValueError:
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True)
        return contrib[:, :-1], None
