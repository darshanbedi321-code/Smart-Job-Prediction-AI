# natural language explainer
# importing libraries
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)


def generate_explanation(result, user_skills):
    if client is None:
        return ("error"
        )
    predicted_role = result["predicted_role"]
    predicted_salary = result["predicted_salary"]
    top_companies_df = result["top_companies"]

    companies_text = ""
    for _, row in top_companies_df.iterrows():
        companies_text += (
            f"- {row['company']} (Rating: {row['avg_rating']}, "
            f"Avg Salary: ₹{row['avg_salary']:.0f})\n"
        )

    prompt = f"""You are a friendly career advisor who understands the Indian tech job market.

User's skills: {user_skills}

Analysis results:
- Predicted job role: {predicted_role}
- Predicted average salary: Rs {predicted_salary:.0f} per year
- Top matching companies:
{companies_text}

Based on this data, write a short, friendly, encouraging explanation in **English only** (150-200 words) that covers:
1. Why this role suits them based on their skills
2. Context on the salary (is it good, average, or could it improve)
3. Which of the top companies seems best and why
4. One small actionable tip (e.g. adding a particular skill would open up better opportunities)

Use a simple, conversational tone, as if a friend is giving advice.
Do not use any language other than English."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"erro: {e}"
