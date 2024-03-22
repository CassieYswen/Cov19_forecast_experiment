#import openai
import os
import openai

#set api key as environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

#check if the api_key is loaded succefssfully
#print(f"Secret key: {openai.api_key}")

#openai.base_url = "https://..."
openai.default_headers = {"x-foo": "true"}
# Define a function to call the OpenAI GPT-4 API
def ask_gpt(prompt, temperature):
     completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=temperature,  # Setting the temperature
        )
     return completion.choices[0].message['content']

text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = ask_gpt(prompt,temperature=0.7)
print(response)