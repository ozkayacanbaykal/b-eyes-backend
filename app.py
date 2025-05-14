from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/analyze", methods=["POST"])
def analyze_bias():
    data = request.json
    text = data.get("text", "")
    print("📥 [DEBUG] Received text (first 300 chars):", text[:300])


    messages = [
        {
            "role": "system",
            "content": """You are an expert media analyst evaluating the following article for bias, manipulation, and credibility.

Return a JSON object structured as follows:

{
  "final_score": (integer between 0–100),
  "categories": {
    "FactualAccuracy": { "score": ..., "explanation": "..." },
    "Objectivity": { "score": ..., "explanation": "..." },
    "SourceQuality": { "score": ..., "explanation": "..." },
    "Language": { "score": ..., "explanation": "..." }
  },
  "penalties": [ { "type": "...", "count": ..., "points_lost": ... }, ... ],
  "flagged_phrases": [
    {
      "text": "...",
      "explanation": {
        "why": "...",
        "type": "...",
        "fix": "..."
      }
    },
    ...
  ]
}

🔎 Carefully scan the **entire text**. Identify and return **all notable biased or manipulative phrases** (aim for 3–7 or more if applicable). These can include:
- emotionally loaded language
- misleading framing
- one-sided statements
- exaggeration or omission
- demagoguery or propaganda

For each flagged phrase:
• Explain clearly *why* it's biased
• Classify its *type* (e.g., "loaded language", "framing", etc.)
• Suggest how to *fix* or rephrase it neutrally

The final_score should be calculated using:
(FactualAccuracy × 0.4) + (Objectivity × 0.3) + (SourceQuality × 0.2) + (Language × 0.1) - penalty points

Even if the article is mostly neutral, return a minimal penalty and short explanation."""
        },
        {
            "role": "user",
            "content": f"Analyze this article:\n\"\"\"{text}\"\"\""
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            max_tokens=1500
        )
        content = response.choices[0].message.content
        print("🔎 GPT Response:\n", content)

        try:
            result = eval(content)
            return jsonify(result)
        except Exception as e:
            print("❌ Failed to parse GPT response:", e)
            return jsonify({"error": "Parsing error", "raw": content}), 500

    except Exception as e:
        print("❌ OpenAI request failed:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data.get("text", "")

    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant. Summarize the article with:

1. **Summary:** A few plain English sentences.
2. **Key Points:** Bullet points with key information.
3. **Takeaways:** 2–3 memorable insights or lessons.

Use Markdown for bold and bullets. Keep it readable."""
        },
        {
            "role": "user",
            "content": text
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            max_tokens=800
        )
        summary = response.choices[0].message.content
        return jsonify({"summary": summary})
    except Exception as e:
        print("❌ OpenAI request failed:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
