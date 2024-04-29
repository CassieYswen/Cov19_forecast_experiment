# Import necessary libraries
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from docx import Document

# Function to read text from a Word document
def read_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def generate_and_save_word_cloud(text, output_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()

# Directory containing the reports
reports_dir = "C:\\Users\\weny\\Dropbox\\__Dr.B\\reports"
output_dir = "outputs/blogs"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each document in the reports directory
for filename in os.listdir(reports_dir):
    if filename.endswith(".docx"):
        file_path = os.path.join(reports_dir, filename)
        blog_text = read_text_from_docx(file_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_wordcloud.png")
        generate_and_save_word_cloud(blog_text, output_path)
