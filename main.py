# installing main libraries
import streamlit as st
import requests
# backend url
FLASK_API_URL ="https://smart-job-prediction-ai.onrender.com/analyze"
st.set_page_config(
    page_title="Smart Job Market Analyzer",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded",
)
# styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9aa0a6;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.8rem;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid rgba(150,150,150,0.25);
        padding-bottom: 0.4rem;
    }
    .company-card {
        background-color: rgba(120,120,120,0.08);
        border: 1px solid rgba(150,150,150,0.2);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .company-name {
        font-weight: 600;
        font-size: 1.02rem;
    }
    .company-meta {
        color: #9aa0a6;
        font-size: 0.9rem;
        margin-top: 0.15rem;
    }
    .rank-badge {
        display: inline-block;
        background-color: #2b6cb0;
        color: white;
        border-radius: 6px;
        padding: 0.05rem 0.55rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .explanation-box {
        background-color: rgba(43, 108, 176, 0.08);
        border-left: 4px solid #2b6cb0;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# side about section
with st.sidebar:
    st.header("About")
    st.write(
        "This tool predicts your best-fit job role, expected salary, "
        "and top hiring companies based on your skills — powered by "
        "machine learning models trained on real job market data."
    )
    st.divider()
    st.caption("**Pipeline**")
    st.caption("Random Forest → Job Role")
    st.caption("Gradient Boosting → Salary")
    st.caption("Groq LLM → Explanation")
    st.divider()
    st.caption("Built by Darshan Bedi")
# header
st.markdown('<div class="main-title">💼 Smart Job Market Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter your skills and get an AI-powered career analysis — '
    'predicted role, expected salary, and top companies hiring for you.</div>',
    unsafe_allow_html=True,
)
# skills section
SKILL_CATEGORIES = {
    "Programming Languages": [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "C#",
        "C++",
        "C",
        "R",
    ],

    "Web & Frontend": [
        "React",
        "Angular",
        "AngularJS",
        "HTML",
        "CSS",
        "jQuery",
        "Bootstrap",
        "Redux",
    ],

    "Backend & Frameworks": [
        "Django",
        "Spring Boot",
        "Spring",
        "Hibernate",
        "ASP.NET",
        ".NET",
        "FastAPI",
        "Node.js",
        "Microservices",
        "REST",
    ],

    "Cloud & DevOps": [
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
        "DevOps",
        "Jenkins",
        "Linux",
        "Git",
        "CI/CD",
    ],

    "Databases": [
        "SQL",
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "SQL Server",
        "Oracle",
        "NoSQL",
        "Snowflake",
    ],

    "Data Science & AI/ML": [
        "Machine Learning",
        "Deep Learning",
        "Artificial Intelligence",
        "TensorFlow",
        "PyTorch",
        "Pandas",
        "NumPy",
        "PySpark",
        "Spark",
        "Big Data",
        "Hadoop",
        "Natural Language Processing",
        "Neural Networks",
        "Generative AI",
        "Kafka",
    ],

    "Data Engineering & BI": [
        "Data Engineering",
        "ETL",
        "Data Modeling",
        "Data Warehousing",
        "Power BI",
        "Tableau",
        "Data Analytics",
        "Data Visualization",
    ],

    "Enterprise Tools": [
        "SAP",
        "SAP ABAP",
        "SAP HANA",
        "Salesforce",
        "Workday",
        "ServiceNow",
    ],

    "Mobile Development": [
        "Android",
        "React Native",
        "Kotlin",
        "Swift",
        "Flutter",
    ],
}
# user skills collection section
with st.form("analyze_form"):
    st.write("**Select Your Skills**")

    selected_skills = []
    for category, options in SKILL_CATEGORIES.items():
        chosen = st.multiselect(category, options)
        selected_skills.extend(chosen)

    custom_skills = st.text_input(
        "Other skills (optional, comma-separated)",
        placeholder="e.g. Blockchain, GraphQL",
    )

    col1, col2 = st.columns(2)
    with col1:
        experience = st.slider("Years of Experience", min_value=0, max_value=15, value=3)
    with col2:
        rating_pref = st.slider("Preferred Company Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

    if custom_skills.strip():
        selected_skills.extend([s.strip() for s in custom_skills.split(",") if s.strip()])
    skills_input = ", ".join(selected_skills)

    if selected_skills:
        st.caption(f"Selected ({len(selected_skills)}): {skills_input}")

    submitted = st.form_submit_button("🔍 Analyze My Career", use_container_width=True)
# backend calling 
if submitted:
    if not skills_input.strip():
        st.error("Please select at least one skill before analyzing.")
    else:
        with st.spinner("Analyzing your profile against the job market..."):
            try:
                response = requests.post(
                    FLASK_API_URL,
                    json={
                        "skills": skills_input,
                        "experience": experience,
                        "rating_pref": rating_pref,
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()

                    st.success("Analysis complete!")

                    #Predicted Role + Salary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Role", data["predicted_role"])
                    with col2:
                        st.metric("Expected Salary", f"{data['predicted_salary']:,.0f} / yr")

                    #Explanation
                    st.markdown('<div class="section-header"> AI Career Insight</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="explanation-box">{data["explanation"]}</div>',
                        unsafe_allow_html=True,
                    )

                    #Top Companies
                    st.markdown('<div class="section-header"> Top Hiring Companies</div>', unsafe_allow_html=True)
                    companies = data.get("top_companies", [])

                    if companies:
                        for i, c in enumerate(companies, start=1):
                            st.markdown(
                                f"""
                                <div class="company-card">
                                    <span class="rank-badge">#{i}</span>
                                    <span class="company-name">{c['company']}</span>
                                    <div class="company-meta">
                                         {c['avg_rating']:.1f} rating &nbsp;|&nbsp;
                                         rs {c['avg_salary']:,.0f} avg salary &nbsp;|&nbsp;
                                         {c['job_count']} open role(s)
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No company data available for this role yet.")

                else:
                    st.error(f"API error: {response.json().get('error', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "error occur"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
# credits and disclaimer
st.divider()
st.caption("Data sourced from AmbitionBox · Predictions are estimates, not guarantees.")
