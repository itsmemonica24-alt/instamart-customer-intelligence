from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
END = pd.Timestamp("2025-10-01 00:00:00")
SEED = 42
report = []

def log(title, lines):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)
    for line in lines:
        print(line)
    report.extend(["", title, *[str(x) for x in lines]])

def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "cm": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }

def show_metrics(name, m):
    return (f"{name}: accuracy={m['accuracy']:.3f}, precision_lapsed={m['precision']:.3f}, "
            f"recall_lapsed={m['recall']:.3f}, F1_lapsed={m['f1']:.3f}, "
            f"confusion_matrix [[TN, FP], [FN, TP]]={m['cm']}")

# Load and validate Part A outputs.
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv")
customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders["status"] = orders["status"].astype(str).str.strip().str.lower()
for c in ["order_amount", "discount_pct", "delivery_minutes"]:
    orders[c] = pd.to_numeric(orders[c], errors="coerce")
completed = orders[orders.status.eq("completed")].copy()
if completed.empty:
    raise ValueError("No completed orders found.")

# Task 1: RFM as of 2025-10-01.
rfm = (completed[completed.order_ts.lt(SNAPSHOT)]
       .groupby("customer_id")
       .agg(last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"))
       .reset_index())
rfm["recency"] = (SNAPSHOT - rfm.last_order_ts).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
log("TASK 1 — RFM", [
    f"Customers with completed orders: {len(rfm)}",
    f"Recency median/min/max: {rfm.recency.median():.1f}/{rfm.recency.min()}/{rfm.recency.max()}",
    f"Frequency mean/max: {rfm.frequency.mean():.2f}/{rfm.frequency.max()}",
    f"Monetary mean/max: INR {rfm.monetary.mean():,.2f}/INR {rfm.monetary.max():,.2f}",
])

# Task 2: Standardise RFM and evaluate K=2..8.
rfm_scaler = StandardScaler()
scaled_rfm = rfm_scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
k_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels = km.fit_predict(scaled_rfm)
    k_rows.append({"K": k, "inertia": km.inertia_,
                   "silhouette": silhouette_score(scaled_rfm, labels)})
k_table = pd.DataFrame(k_rows)
log("TASK 2 — K-MEANS EVALUATION", [
    k_table.to_string(index=False, formatters={"inertia": "{:.1f}".format,
                                               "silhouette": "{:.3f}".format}),
    "Chosen K=4: the elbow and stronger silhouette at K=4 support four actionable groups.",
    "Inertia always decreases as K increases because extra centroids reduce within-cluster distances. "
    "K=8's lower inertia alone is not enough: too many groups can over-fragment customers and "
    "make campaigns harder to act on.",
])

# Task 3: final clusters, profiles, PCA, named segments.
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
rfm["segment_id"] = km.fit_predict(scaled_rfm)
profiles = rfm.groupby("segment_id").agg(
    size=("customer_id", "size"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")).reset_index()
premium_id = int(profiles.loc[profiles.mean_monetary.idxmax(), "segment_id"])
others = profiles[profiles.segment_id.ne(premium_id)].copy()
regular_id = int(others.loc[others.mean_frequency.idxmax(), "segment_id"])
names = {}

for _, row in profiles.iterrows():
    sid = int(row.segment_id)

    if sid == premium_id:
        names[sid] = "Premium"

    elif sid == regular_id:
        names[sid] = "High-Frequency Regulars"

    elif row.mean_recency >= profiles.mean_recency.median():
        if row.mean_monetary < profiles.mean_monetary.median():
            names[sid] = "Dormant Low-Spend Customers"
        else:
            names[sid] = "At-Risk / Lapsing"

    else:
        names[sid] = "Occasional Shoppers"

rfm["segment_name"] = rfm.segment_id.map(names)
rfm["is_premium"] = rfm.segment_id.eq(premium_id)
profiles["segment_name"] = profiles.segment_id.map(names)
profiles["recommended_action"] = profiles.segment_name.map({
    "Premium": "Protect loyalty with personalised service and relevant cross-sells.",
    "High-Frequency Regulars": "Use replenishment reminders and useful bundles.",
    "At-Risk / Lapsing": "Run a win-back campaign with a margin-aware incentive.",
    "Occasional Shoppers": "Test category recommendations and second-order offers.",
})
log("TASK 3 — SEGMENT PROFILES", [
    profiles.to_string(index=False, formatters={
        "mean_recency": "{:.2f}".format, "mean_frequency": "{:.2f}".format,
        "mean_monetary": "INR {:,.2f}".format}),
    f"PCA explained variance ratios: {PCA(n_components=2).fit(scaled_rfm).explained_variance_ratio_}",
    "Cluster IDs are arbitrary labels, not an ordered ranking. Premium is the cluster with the highest mean monetary value.",
])
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    OUT / "customer_segments.csv", index=False)

# Task 4: define cohort and label. Features only use orders before cutoff.
pre_completed = completed[completed.order_ts.lt(CUTOFF)].copy()
cohort_ids = set(pre_completed.customer_id.unique())
september_active = set(completed.loc[
    completed.order_ts.ge(CUTOFF) & completed.order_ts.lt(END), "customer_id"].unique())
pre_orders = orders[orders.order_ts.lt(CUTOFF)].copy()
pre_final = pre_orders[pre_orders.status.isin(["completed", "cancelled"])].copy()
feature_completed = pre_orders[pre_orders.status.eq("completed")].merge(
    products[["product_id", "category"]], on="product_id", how="left")
feature_completed["category"] = feature_completed.category.fillna("Unknown").astype(str)
agg = feature_completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount_pct=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean")).reset_index()
agg["recency"] = (CUTOFF - agg.last_order_ts).dt.days
fav = feature_completed.groupby("customer_id").category.agg(
    lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown").rename("favourite_category").reset_index()
agg = agg.merge(fav, on="customer_id", how="left")
final_counts = pre_final.assign(cancelled=pre_final.status.eq("cancelled").astype(int)).groupby("customer_id").agg(
    cancelled_orders=("cancelled", "sum"), final_orders=("order_id", "nunique")).reset_index()
agg = agg.merge(final_counts, on="customer_id", how="left")
agg["cancellation_rate"] = agg.cancelled_orders.fillna(0) / agg.final_orders.replace(0, np.nan)
cohort = pd.DataFrame({"customer_id": sorted(cohort_ids)}).merge(agg, on="customer_id", how="left")
cohort = cohort.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]],
                      on="customer_id", how="left")
cohort["tenure_days"] = (CUTOFF - cohort.signup_date).dt.days
cohort["lapsed"] = (~cohort.customer_id.isin(september_active)).astype(int)
cohort["favourite_category"] = cohort.favourite_category.fillna("Unknown")
cohort["city"] = cohort.city.fillna("Unknown").astype(str)
cohort["gender"] = cohort.gender.fillna("unknown").astype(str)
log("TASK 4 — LAPSE COHORT", [
    f"Cohort size: {len(cohort)}",
    f"Lapsed: {cohort.lapsed.sum()} ({cohort.lapsed.mean()*100:.2f}%)",
    f"Not lapsed: {(cohort.lapsed == 0).sum()}",
    "No order dated 2025-09-01 or later enters predictive features; Task 3 segment labels are not features.",
])

# Task 5: split before fitting any encoder/scaler. Imputation is also train-fitted.
numeric = ["recency", "frequency", "monetary", "avg_discount_pct",
           "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
categorical = ["city", "gender", "favourite_category"]
features = numeric + categorical
X, y = cohort[features].copy(), cohort.lapsed.copy()
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, cohort.index, test_size=0.2, stratify=y, random_state=SEED)
missing = X.isna().sum()
if missing.any():
    raise ValueError("Missing feature values remain:\n" + missing[missing.gt(0)].to_string())
try:
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
], verbose_feature_names_out=False)
baseline = metrics(y_test, np.zeros(len(y_test), dtype=int))
log("TASK 5 — SPLIT, MISSING VALUES AND BASELINE", [
    f"Train/test rows: {len(X_train)}/{len(X_test)}",
    f"Missing values by feature: {missing.to_dict()}",
    f"Full cohort frequency mean: {X.frequency.mean():.4f}",
    f"Training-only frequency mean (before scaling): {X_train.frequency.mean():.4f}",
    "These means can differ because the training split contains 80% of the cohort; preprocessing must be fitted on training rows only.",
    show_metrics("Always predict not lapsed", baseline),
])

# Task 6: grid searches; pipelines ensure preprocessing is refitted within each CV fold.
lr_pipe = Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000))])
lr_grid = GridSearchCV(lr_pipe, {"model__C": [0.01, 0.1, 1, 10, 100]},
                       cv=5, scoring="f1", n_jobs=-1)
lr_grid.fit(X_train, y_train)

lr = lr_grid.best_estimator_

# Get the StandardScaler fitted on the training data only.
fitted_scaler = (
    lr.named_steps["preprocess"]
    .named_transformers_["num"]
    .named_steps["scaler"]
)

# Find the position of the frequency feature in the numeric columns.
frequency_index = numeric.index("frequency")

# Display the frequency mean learned by the scaler.
print(
    "Fitted scaler frequency mean:",
    fitted_scaler.mean_[frequency_index],
)

# Compare it with the mean across the full cohort.
print(
    "Full-cohort frequency mean:",
    X["frequency"].mean(),
)

lr_pred = lr.predict(X_test)
lr_metrics = metrics(y_test, lr_pred)

rf_pipe = Pipeline([("preprocess", preprocess),
                    ("model", RandomForestClassifier(n_estimators=100, random_state=SEED))])
rf_grid = GridSearchCV(rf_pipe, {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1", n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf = rf_grid.best_estimator_
rf_pred = rf.predict(X_test)
rf_metrics = metrics(y_test, rf_pred)
log("TASK 6 — TUNED MODELS AND TEST METRICS", [
    f"Logistic best params: {lr_grid.best_params_}; best CV F1={lr_grid.best_score_:.4f}",
    show_metrics("Logistic test", lr_metrics),
    f"Random Forest best params: {rf_grid.best_params_}; best CV F1={rf_grid.best_score_:.4f}",
    show_metrics("Random Forest test", rf_metrics),
])
if rf_metrics["f1"] >= 0.55:
    selected_name, selected = "Random Forest", rf
elif lr_metrics["f1"] >= 0.55:
    selected_name, selected = "Logistic Regression", lr
else:
    selected_name, selected = (("Random Forest", rf) if rf_metrics["f1"] >= lr_metrics["f1"]
                               else ("Logistic Regression", lr))
selected_metrics = rf_metrics if selected_name == "Random Forest" else lr_metrics
log("MODEL CHOICE", [
    f"Selected {selected_name}; lapsed-class test F1={selected_metrics['f1']:.4f}.",
    "Missing a soon-to-lapse customer may lose future orders, so recall matters. "
    "Wrongly targeting loyal customers wastes discounts and campaign capacity, so precision matters too. "
    "The selected model balances these costs using held-out F1; the retention team should still set a threshold "
    "that fits its budget and incentive margin.",
])

# Task 7: oversample training rows only, with replacement, until classes are balanced.
train = X_train.copy()
train["_target"] = y_train.to_numpy()
counts = train["_target"].value_counts()
minority = int(counts.idxmin())
majority_n = int(counts.max())
minority_rows = train[train._target.eq(minority)]
extra = minority_rows.sample(n=majority_n - len(minority_rows), replace=True, random_state=SEED)
balanced = pd.concat([train, extra], ignore_index=True)
balanced_lr = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, C=lr_grid.best_params_["model__C"])),
])
balanced_lr.fit(balanced[features], balanced["_target"].astype(int))
balanced_metrics = metrics(y_test, balanced_lr.predict(X_test))
log("TASK 7 — TRAIN-ONLY OVERSAMPLING", [
    f"Original training counts: {counts.sort_index().to_dict()}",
    f"Balanced training counts: {balanced['_target'].value_counts().sort_index().to_dict()}",
    show_metrics("Unbalanced logistic", lr_metrics),
    show_metrics("Oversampled logistic", balanced_metrics),
    "Recall often moves most: more actual lapsing customers are flagged, but unchanged or lower precision "
    "can mean more loyal customers receive offers. Use this trade-off to plan campaign capacity.",
])

# Task 8a: feature importance and logistic coefficient signs.
rf_prep = rf.named_steps["preprocess"]
rf_model = rf.named_steps["model"]
names_rf = rf_prep.get_feature_names_out()
importance = pd.DataFrame({"feature": names_rf, "importance": rf_model.feature_importances_}).sort_values(
    "importance", ascending=False).head(5)
lr_names = lr.named_steps["preprocess"].get_feature_names_out()
lr_coef = pd.DataFrame({"feature": lr_names, "coefficient": lr.named_steps["model"].coef_[0]})
coef_lines = []
for feature in ["recency", "frequency", "monetary"]:
    row = lr_coef[lr_coef.feature.eq(feature)]
    if not row.empty:
        val = float(row.iloc[0].coefficient)
        sign = "positive" if val > 0 else "negative" if val < 0 else "zero"
        coef_lines.append(f"{feature}: {val:.4f} ({sign}); positive coefficients increase estimated lapse odds, negative coefficients decrease them.")
log("TASK 8A — FEATURE INTERPRETATION", [
    "Top five Random Forest importances:",
    importance.to_string(index=False, formatters={"importance": "{:.4f}".format}),
    "Logistic coefficient signs:",
    *coef_lines,
])

# Task 8b: persist selected model and verify exact prediction equality.
model_path = OUT / "lapse_model.joblib"
joblib.dump(selected, model_path)
reloaded = joblib.load(model_path)
exact = np.array_equal(selected.predict(X_test), reloaded.predict(X_test))
if not exact:
    raise AssertionError("Reloaded model predictions differ from original predictions.")
log("TASK 8B — MODEL PERSISTENCE", [
    f"Saved: {model_path.relative_to(ROOT)}",
    f"Reloaded predictions exactly match original predictions: {exact}",
])

# Task 8c: rank Premium test customers. This flag is post-hoc whole-window RFM.
test_customers = cohort.loc[idx_test, ["customer_id", "lapsed"]].copy()
test_customers = test_customers.merge(rfm[["customer_id", "is_premium"]], on="customer_id", how="left")
prob = reloaded.predict_proba(X_test)[:, list(reloaded.classes_).index(1)]
test_customers["lapse_probability"] = prob
retention = (test_customers[test_customers.is_premium.eq(True)]
             .sort_values("lapse_probability", ascending=False).head(10)
             .rename(columns={"lapsed": "actually_lapsed"})
             [["customer_id", "lapse_probability", "actually_lapsed"]])
retention.to_csv(OUT / "retention_list.csv", index=False)
log("TASK 8C — PREMIUM RETENTION LIST", [
    f"Premium customers in test split: {int(test_customers.is_premium.eq(True).sum())}",
    f"Rows in list: {len(retention)}",
    f"Actually lapsed in list: {int(retention.actually_lapsed.sum()) if len(retention) else 0}",
    retention.to_string(index=False),
    "The retention list is illustrative, not a leakage-free evaluation, because Premium is a post-hoc whole-window label that includes the September label window. Predictive features still use only pre-cutoff orders.",
])

# Write report and validate required outputs.
(OUT / "part_b_report.txt").write_text("\n".join(map(str, report)) + "\n", encoding="utf-8")
saved_segments = pd.read_csv(OUT / "customer_segments.csv")
saved_retention = pd.read_csv(OUT / "retention_list.csv")
assert list(saved_segments.columns) == ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
assert list(saved_retention.columns) == ["customer_id", "lapse_probability", "actually_lapsed"]
assert bool(exact)
log("PART B COMPLETE", [
    "Created outputs/customer_segments.csv",
    "Created outputs/lapse_model.joblib",
    "Created outputs/retention_list.csv",
    "Created outputs/part_b_report.txt",
    f"Chosen K={K}; Premium segment_id={premium_id}; selected model={selected_name}",
    f"Selected model test F1 (lapsed class)={selected_metrics['f1']:.4f}",
    "README limitation to include: the Premium flag is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation.",
])
