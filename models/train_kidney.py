import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("datasets/kidney_disease.csv")

df = df.drop(columns=["id"])

# Clean target column
df["classification"] = df["classification"].str.strip()
df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})

# Columns that are text/categorical
categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

# Some numeric columns got read as text due to stray characters — force numeric
numeric_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
                 "hemo", "pcv", "wc", "rc"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing values
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.dropna(subset=["classification"])

# Encode categorical text to numbers
mappings = {
    "rbc": {"normal": 1, "abnormal": 0},
    "pc": {"normal": 1, "abnormal": 0},
    "pcc": {"present": 1, "notpresent": 0},
    "ba": {"present": 1, "notpresent": 0},
    "htn": {"yes": 1, "no": 0},
    "dm": {"yes": 1, "no": 0},
    "cad": {"yes": 1, "no": 0},
    "appet": {"good": 1, "poor": 0},
    "pe": {"yes": 1, "no": 0},
    "ane": {"yes": 1, "no": 0},
}
for col, mapping in mappings.items():
    df[col] = df[col].map(mapping)
    df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop(columns=["classification"])
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Kidney Model Accuracy:", accuracy_score(y_test, preds))

joblib.dump(model, "saved_models/kidney_model.pkl")
joblib.dump(scaler, "saved_models/kidney_scaler.pkl")
print("Columns used:", X.columns.tolist())