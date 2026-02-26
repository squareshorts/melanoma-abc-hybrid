import shap

def shap_values_tree(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    return sv, explainer
