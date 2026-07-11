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

    prompt = f"""Tum ek friendly career advisor ho jo Indian tech job market samajhte ho.

User ki skills: {user_skills}

Analysis ke results:
- Predicted job role: {predicted_role}
- Predicted average salary: rs{predicted_salary:.0f} per year
- Top matching companies:
{companies_text}

Is data ke basis pe user ko ek short, friendly, encouraging explanation do (150-200 words mein) jisme:
1. Unki skills ke basis pe ye role kyun suit karta hai
2. Salary ka context (ye achha hai, average hai, ya improve ho sakta hai)
3. Top companies mein se konsi sabse achhi lagti hai aur kyun
4. Ek chhota actionable tip (jaise koi skill add karo to better opportunities milengi)

Simple, conversational tone use karo, jaise ek dost advice de raha ho."""

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
