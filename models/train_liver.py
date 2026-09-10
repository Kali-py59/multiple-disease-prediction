import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("datasets/indian_liver_patient.csv")

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Fill the one column known to have missing values
df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
    df["Albumin_and_Globulin_Ratio"].median()
)

# Dataset column: 1 = liver disease, 2 = no disease -> convert to 1/0
df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})

X = df.drop(columns=["Dataset"])
y = df["Dataset"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Liver Model Accuracy:", accuracy_score(y_test, preds))

joblib.dump(model, "saved_models/liver_model.pkl")
joblib.dump(scaler, "saved_models/liver_scaler.pkl")
print("Columns used:", X.columns.tolist())