import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("data.csv")

print("\n===== FIRST 5 RECORDS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ==========================
# HANDLE MISSING VALUES
# ==========================

df = df.ffill()

# ==========================
# ENCODE CATEGORICAL COLUMNS
# ==========================

encoder = LabelEncoder()

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = encoder.fit_transform(df[col].astype(str))

# ==========================
# FIND TARGET COLUMN
# ==========================

target = "Exam_Score"

if target not in df.columns:
    print("\nAvailable Columns:")
    print(df.columns)
    raise Exception(
        "Column 'Exam_Score' not found. Check the column name in your dataset."
    )

# ==========================
# BASIC ANALYSIS
# ==========================

print("\n===== EXAM SCORE ANALYSIS =====")

print("Average Score :", df[target].mean())
print("Highest Score :", df[target].max())
print("Lowest Score :", df[target].min())

# ==========================
# CORRELATION
# ==========================

print("\n===== CORRELATION WITH EXAM SCORE =====")

corr = df.corr(numeric_only=True)[target].sort_values(ascending=False)

print(corr)

# ==========================
# MACHINE LEARNING
# ==========================

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")

print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("R2 Score :", round(r2, 4))

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

# ==========================
# SAMPLE PREDICTIONS
# ==========================

results = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": predictions
})

print("\n===== SAMPLE PREDICTIONS =====")
print(results.head(10))

results.to_csv(
    "sample_predictions.csv",
    index=False
)

print("\n===== FILES CREATED =====")
print("feature_importance.csv")
print("sample_predictions.csv")

print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")