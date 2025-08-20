from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import re

app = FastAPI()
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

class FactCheckRequest(BaseModel):
    text: str

@app.post("/fact-check")
async def fact_check(req: FactCheckRequest):
    # 1. Split text into sentences/claims
    claims = [c.strip() for c in re.split(r'[.!?]', req.text) if c.strip()]

    results = []
    for claim in claims:
        prompt = f"""
        You are a fact-checking assistant.
        Claim: "{claim}"

        Task:
        1. Decide if this claim is True, False, or Unverifiable.
        2. Provide a short explanation.
        3. Suggest possible sources.

        Answer in JSON:
        {{
          "claim": "{claim}",
          "verdict": "True/False/Unverifiable",
          "explanation": "Reason",
          "sources": ["Source1", "Source2"]
        }}
        """

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        # Parse response
        results.append(response.choices[0].message.content)

    return {"results": results}
