# backend of the project
from flask import Flask, request, jsonify
from model_utils import analyze_career
from llm_utils import generate_explanation
# making object
app = Flask(__name__)
# server checking
@app.route("/health", methods=["GET"])
def health():
    """Lord Darshan"""
    return jsonify({"status": "ok", "message": "Server is working properly"}), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)

    if not data or "skills" not in data or not data["skills"].strip():
        return jsonify({"error": "fill data first"}), 400

    user_skills = data["skills"]
    avg_exp = data.get("experience", 3)
    rating_pref = data.get("rating_pref", 4.0)

    try:
        # first models data predict kraga
        result = analyze_career(user_skills, avg_exp_input=avg_exp, rating_input=rating_pref)

        #natural language explanation 
        explanation = generate_explanation(result, user_skills)

        #response formating to json 
        response = {
            "predicted_role": result["predicted_role"],
            "predicted_salary": result["predicted_salary"],
            "top_companies": result["top_companies"].to_dict(orient="records"),
            "explanation": explanation,
        }

        return jsonify(response), 200

    except Exception as e:
        #error handling
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
