import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello, Groq!"}],
    model="llama-3.3-70b-versatile"
)
print(response.choices[0].message.content)
