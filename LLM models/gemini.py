import pathlib
import textwrap
import os   
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


  #pick up API key in the environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-pro-latest')

prompt="What is the meaning of life?"

response = model.generate_content(prompt)

print(response.text)

# Ensure the directory exists
output_dir = 'outputs/gemini'
os.makedirs(output_dir, exist_ok=True)

# Path for the file where the prompt and response will be saved
file_path = os.path.join(output_dir, 'prompt_and_response.txt')

# Write the prompt and response to a text file
with open(file_path, 'w') as file:
    file.write("Prompt: " + prompt + "\n")
    file.write("Response: " + response.text + "\n")

print(f"Prompt and response have been saved to {file_path}")