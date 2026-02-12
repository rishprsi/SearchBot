import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if api_key is None:
    raise ValueError("Please provide an api key before proceeding")
print(f"Using key {api_key[:6]}...")
client = genai.Client(api_key=api_key)

prompt = (
    "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
)

content = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
print(content.text)
if content is None or content.usage_metadata is None:
    print("No metadata")
else:
    print("Prompt Tokens: ", content.usage_metadata.prompt_token_count)
    print("Response Tokens: ", content.usage_metadata.candidates_token_count)
