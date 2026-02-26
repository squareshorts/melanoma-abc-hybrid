import joblib

def main():

    artifact = joblib.load("results/runs/hybrid/hybrid_xgb.joblib")
    model = artifact["model"]

    booster = model.get_booster()

    # Save in stable native format
    booster.save_model("results/runs/hybrid/hybrid_xgb_native.json")

    print("Model re-exported to native JSON format.")


if __name__ == "__main__":
    main()