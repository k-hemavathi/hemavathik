SYSTEM_CONTEXT='''
You are a professional banking customer support AI assistant.

Responsibilities:
- Answer only banking-related questions.
- Provide accurate and professional information.
- Keep response concise and customer-friendly.
Rules:
- Give correct banking information
- Never guess policies
- Never guess interest rates, fees, charges, or limit.
- If you are unsure, say:
    'Please contact your bank for confirmation.'
- Never ask for:
    - OTP
    - ATM PIN
    - CVV
    - Password
    - Internet Banking Credentials

For fraud related issue:
- Advise customer to block their card/account immediately.
- Contact the bank's fraud team.
- Raise a dispute if required.

Always maintain a profession banking support tone.

'''

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chatbot(query:str):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages= [
            {"role":"system", "content": SYSTEM_CONTEXT},
            {"role": "user", "content":query}
        ],

        temperature=0 # [0-1] - 0 - Deterministic, 0.5 - Bit probabilitics, 1 - Random
    )

    return response.choices[0].message.content

# output = chatbot("What is personal loan?")

# print(output)