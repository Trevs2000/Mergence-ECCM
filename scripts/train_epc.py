import pandas as pd
from sklearn.model_selection import train_test_split
from metrics.epc import EPCTrainer

# Load all merge results
fraud_results = pd.read_csv("./results/merges/fraud/merge_results_new.csv")
churn_results = pd.read_csv("./results/merges/churn/merge_results_new.csv")

# Combine fraud + churn merge history
all_merges = pd.concat([fraud_results, churn_results], ignore_index=True)
print(f"Training EPC on {len(all_merges)} merge experiments...\n")

# 1) Train/test split for proper evaluation
train_merges, test_merges = train_test_split(
    all_merges, test_size=0.2, random_state=42
)

# 2) Train EPC on train_merges only
epc = EPCTrainer()
train_r2 = epc.train(train_merges, n_trees=100)
print(f"Train R²: {train_r2:.4f}")

# 3) Evaluate on held‑out test_merges
X_test = test_merges[["psc", "fsc", "rsc"]].values
y_test = test_merges["improvement"].values
test_r2 = epc.model.score(X_test, y_test)
print(f"Test R²: {test_r2:.4f}")

# 4) Feature importances and normalised ECCM weights
importances = epc.model.feature_importances_
print("\nFeature Importances:")
print(f"  PSC: {importances[0]:.3f}")
print(f"  FSC: {importances[1]:.3f}")
print(f"  RSC: {importances[2]:.3f}")

total = importances.sum()
w_psc = importances[0] / total
w_fsc = importances[1] / total
w_rsc = importances[2] / total
print(f"\nOptimised ECCM weights:")
print(f"  w_PSC = {w_psc:.3f}")
print(f"  w_FSC = {w_fsc:.3f}")
print(f"  w_RSC = {w_rsc:.3f}")

# 5) Save trained EPC model
epc.save("./models/epc_model.pkl")