# importing basic libraries 
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time
import os
# webscaping from ambitionbox
BASE_URL = (
    "https://www.ambitionbox.com/jobs?campaign=desktop_nav&designation="
    "software-engineer,senior-soft-engineer,data-engineer,full-stack-developer,"
    "java-developer,php-developer,java-full-stack-developer,data-scientist,"
    "android-developer,ai-engineer,python-developer,application-developer,"
    "ai-ml-engineer,web-developer,full-stack-engineer,senior-data-engineer,"
    "front-end-developer,data-analyst,senior-artificial-intelligence-engineer,"
    "reactjs-developer,react-native-developer,software-development-engineer"
)
# so that website give our data easily by thinking we are human instead of spammer or bot
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
# no of pages we scrap and path of both csv's
MAX_PAGES = 500        
RAW_CSV_PATH = "data/ambitionbox_all_jobs.csv"
CLEANED_CSV_PATH = "data/ambitionbox_categorized.csv"
# making catagories
MERGE_MAP = {
    "Senior Software Engineer": "Software Engineer",
    "Software Development Engineer": "Software Engineer",
    "Java Full Stack Developer": "Full Stack Engineer",
    "Full Stack Developer": "Full Stack Engineer",
    "Senior Data Engineer": "Data Engineer",
    "Senior Artificial Intelligence Engineer": "Ai Ml Engineer",
    "AI Engineer": "Ai Ml Engineer",
    "Java Developer": "Application Developer",
    "Front end Developer": "Web Developer",
}

MIN_CATEGORY_COUNT = 10
# appling loop for 
def scrape_jobs():
    all_jobs = []
    page = 1

    while page <= MAX_PAGES:
        url = f"{BASE_URL}&page={page}"
        r = requests.get(url, headers=HEADERS)

        if r.status_code != 200:
            print(f"error on Page no.{page}")
            break

        soup = BeautifulSoup(r.text, "html.parser")
        next_data = soup.find("script", id="__NEXT_DATA__")

        if not next_data:
            print(f"not getting data on Page no.{page}")
            break

        data = json.loads(next_data.string)
        jobs_list = data.get("props", {}).get("pageProps", {}).get("jobsList", [])

        if not jobs_list:
            print(f"nothing is scaped on  Page no.{page}")
            break
# storing data in list 
        for job in jobs_list:
            all_jobs.append({
                "title": job.get("title"),
                "company": job.get("company"),
                "rating": job.get("companyRating"),
                "locations": ", ".join(job.get("locations", [])),
                "min_exp": job.get("minExp"),
                "max_exp": job.get("maxExp"),
                "min_ctc": job.get("minCtc"),
                "max_ctc": job.get("maxCtc"),
                "skills": ", ".join(job.get("skills", [])),
                "job_profile": job.get("jobProfile"),
                "job_profile_id": job.get("jobProfileId"),
            })

        print(f"current Page no.{page}, total data extracted from page is — {len(jobs_list)} ,jobs total data ={len(all_jobs)}")
        page += 1
        time.sleep(2)  
    df = pd.DataFrame(all_jobs)
    df.drop_duplicates(inplace=True)
    return df

# data cleaning
def clean_and_categorize(df):
    df = df.dropna(subset=["min_ctc", "max_ctc"]).copy()
    df["skills"] = df["skills"].fillna("")
    df["avg_ctc"] = (df["max_ctc"] + df["min_ctc"]) / 2
    df["avg_exp"] = (df["max_exp"] + df["min_exp"]) / 2
    df["job_profile"] = df["job_profile"].replace(MERGE_MAP)
    counts = df["job_profile"].value_counts()
    valid_categories = counts[counts >= MIN_CATEGORY_COUNT].index
    df_model = df[df["job_profile"].isin(valid_categories)].copy()

    return df_model

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Scraping shuru...")
    df_raw = scrape_jobs()
    df_raw.to_csv(RAW_CSV_PATH, index=False)
    print(f"\nRaw data saved: {RAW_CSV_PATH} ({len(df_raw)} rows)")

    print("\nCleaning + categorizing...")
    df_clean = clean_and_categorize(df_raw)
    df_clean.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"Cleaned data saved: {CLEANED_CSV_PATH} ({len(df_clean)} rows)")
    print("\nFinal category distribution:")
    print(df_clean["job_profile"].value_counts())
