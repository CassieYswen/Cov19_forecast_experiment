# Import necessary libraries
import os
import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
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

def clean_and_split_text(text):
    # Remove unwanted characters and split text into words
    cleaned_text = ''.join([char.lower() if char.isalnum() or char.isspace() else ' ' for char in text])
    words = cleaned_text.split()
    return words

# Directory containing the reports
reports_dir = "C:\\Users\\weny\\Dropbox\\__Dr.B\\reports"
output_dir = "outputs/blogs"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# # Process each document in the reports directory
# for filename in os.listdir(reports_dir):
#     if filename.endswith(".docx") and not filename.startswith("~$"):
#         file_path = os.path.join(reports_dir, filename)
#         try:    
#             blog_text = read_text_from_docx(file_path)
#             output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_wordcloud.png")
#             generate_and_save_word_cloud(blog_text, output_path)
#         except Exception as e:
#             print(f"Failed to process {filename}: {e}")
#accumulate all text
all_documents_text = ""
#initial a counter to store the word frequency
word_freq = Counter()
#list to store processed file names
processed_files = []

# Process each document in the reports directory
for filename in os.listdir(reports_dir):
    if filename.endswith(".docx") and not filename.startswith("~$"):
        file_path = os.path.join(reports_dir, filename)
        try:
            document_text = read_text_from_docx(file_path)
            all_documents_text += " " + document_text  # Append text from each document
            words = clean_and_split_text(document_text)
            word_freq.update(words)  # Update the Counter with words from this document
            processed_files.append(filename)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# # Generate and save a word cloud for all accumulated text
# output_path = os.path.join(output_dir, "combined_wordcloud.png")
# generate_and_save_word_cloud(all_documents_text, output_path)

# Convert the Counter to a list of (word, frequency) tuples and sort by frequency in descending order
sorted_word_freq = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)

# Save the sorted_word_freq to a .csv file
output_csv_path = os.path.join(output_dir, "word_frequencies.csv")
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Word', 'Frequency'])  # Write the header row
    for word, freq in sorted_word_freq:
        csv_writer.writerow([word, freq])

# # Optionally, print the sorted word frequency list
# for word, freq in sorted_word_freq:
#     print(f"{word}: {freq}")

# Print the processed filenames
print("Processed files:")
for filename in processed_files:
    print(filename)