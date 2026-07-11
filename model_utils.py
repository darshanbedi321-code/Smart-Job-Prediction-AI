# importing libraries
import os
import pandas as pd
import joblib
from scipy.sparse import hstack
# importing data model 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "ambitionbox_categorized.csv")
# loading models and encoders files
job_role_model = joblib.load(os.path.join(MODELS_DIR, "job_role_prediction.pkl"))
tfidf = joblib.load(os.path.join(MODELS_DIR, "skills_vectorization.pkl"))
salary_model = joblib.load(os.path.join(MODELS_DIR, "salary_predictor.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "job_role_encoder.pkl"))

# csv file import krna
df_model = pd.read_csv(DATA_PATH)
# best company predictor
def recommend_companies(predicted_role, df, top_n=5):
    role_jobs = df[df["job_profile"] == predicted_role]

    if role_jobs.empty:
        return pd.DataFrame()

    company_summary = role_jobs.groupby("company").agg(
        avg_rating=("rating", "mean"),
        avg_salary=("avg_ctc", "mean"),
        job_count=("company", "count")
    ).reset_index()

    company_summary = company_summary.sort_values(
        by=["avg_rating", "avg_salary"], ascending=[False, False]
    )
    company_summary["avg_rating"] = company_summary["avg_rating"].round(2)
    company_summary["avg_salary"] = company_summary["avg_salary"].round(0)

    return company_summary.head(top_n)
# job predictor
def analyze_career(user_skills, avg_exp_input=3, rating_input=4.0):
# converting skills into numbers
    skills_vector = tfidf.transform([user_skills])

    # calling job role predicting model
    predicted_role = job_role_model.predict(skills_vector)[0]
    # encoding into string
    role_encoded_input = encoder.transform([[predicted_role]])
    numeric_input = [[avg_exp_input, rating_input]]
    X_input = hstack([skills_vector, role_encoded_input, numeric_input]).toarray()
    predicted_salary = salary_model.predict(X_input)[0]

    # best company recommender
    top_companies = recommend_companies(predicted_role, df_model, top_n=5)
    # giving final result role salary company
    return {
        "predicted_role": str(predicted_role),
        "predicted_salary": round(float(predicted_salary), 2),
        "top_companies": top_companies,
    }
