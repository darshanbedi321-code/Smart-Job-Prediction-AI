# model trainer
# importing libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from scipy.sparse import hstack

RANDOM_STATE = 42
# loading data
df = pd.read_csv("ambitionbox_categorized_clean.csv", keep_default_na=False, na_values=[])
df["skills"] = df["skills"].fillna("")
print(f"Loaded {len(df)} rows, {df['job_profile'].nunique()} job roles")

X_skills = df["skills"]
y_role = df["job_profile"]
# spliting data for train and test
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y_role
)
df_train = df.loc[train_idx].reset_index(drop=True)
df_test = df.loc[test_idx].reset_index(drop=True)
# converting skills into no.
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)
X_train_skills = tfidf.fit_transform(df_train["skills"])
X_test_skills = tfidf.transform(df_test["skills"])
# model and training
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_model.fit(X_train_skills, df_train["job_profile"])
# predicting and accuracy checking
pred_role_test = rf_model.predict(X_test_skills)
role_acc = accuracy_score(df_test["job_profile"], pred_role_test)
print(f"\n=== Job Role Classifier ===")
print(f"Test Accuracy: {role_acc:.4f}")
print(classification_report(df_test["job_profile"], pred_role_test, zero_division=0))
# converting job role into number
encoder = OneHotEncoder(handle_unknown="ignore")
role_train_encoded = encoder.fit_transform(df_train[["job_profile"]])
role_test_encoded = encoder.transform(df_test[["job_profile"]])
# salar model training and predicting and accuracy checking 
X_train_numeric = df_train[["avg_exp", "rating"]].values
X_test_numeric = df_test[["avg_exp", "rating"]].values

X_train_full = hstack([X_train_skills, role_train_encoded, X_train_numeric]).toarray()
X_test_full = hstack([X_test_skills, role_test_encoded, X_test_numeric]).toarray()

salary_model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.08,
    max_depth=8,
    random_state=RANDOM_STATE,
)
salary_model.fit(X_train_full, df_train["avg_ctc"])

pred_salary_test = salary_model.predict(X_test_full)
mae = mean_absolute_error(df_test["avg_ctc"], pred_salary_test)
r2 = r2_score(df_test["avg_ctc"], pred_salary_test)
print(f"\n=== Salary Regressor ===")
print(f"Test MAE: Rs.{mae:,.0f}")
print(f"Test R^2: {r2:.4f}")
# retraining and making final model
tfidf_final = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)
X_all_skills = tfidf_final.fit_transform(df["skills"])

rf_final = RandomForestClassifier(
    n_estimators=150, max_depth=25, min_samples_leaf=2,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf_final.fit(X_all_skills, df["job_profile"])

encoder_final = OneHotEncoder(handle_unknown="ignore")
role_all_encoded = encoder_final.fit_transform(df[["job_profile"]])

X_all_numeric = df[["avg_exp", "rating"]].values
X_all_full = hstack([X_all_skills, role_all_encoded, X_all_numeric]).toarray()

salary_final = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.08, max_depth=8, random_state=RANDOM_STATE
)
salary_final.fit(X_all_full, df["avg_ctc"])
# making files of model and encoder
joblib.dump(tfidf_final, "skills_vectorization.pkl")
joblib.dump(rf_final, "job_role_prediction.pkl")
joblib.dump(encoder_final, "job_role_encoder.pkl")
joblib.dump(salary_final, "salary_predictor.pkl")

print("\nAll models saved.")
