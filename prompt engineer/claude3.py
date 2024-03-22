import anthropic
import os

client = anthropic.Anthropic(
    api_key= os.environ.get("ANTHROPIC_API_KEY"),
    
)
# Define a function to call the claude3 API
def ask_claude(prompt,temperature):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4000,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return message.content

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
response = ask_claude(prompt,temperature=0.7)
print(response)