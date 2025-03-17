import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored  # For highlighting text in terminal

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    pdf_file_obj.close()
    return text

def extract_keywords(text):
    doc = nlp(text)
    keywords = {token.text.lower() for token in doc if not token.is_stop and not token.is_punct}
    return keywords  # Return a set of keywords

def calculate_match(resume_text, jd_text):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    # Find common keywords
    common_keywords = resume_keywords.intersection(jd_keywords)

    # Convert text to numerical vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([" ".join(resume_keywords), " ".join(jd_keywords)])

    # Compute cosine similarity
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    match_percentage = similarity[0][0] * 100

    return match_percentage, common_keywords

def highlight_matched_words(jd_text, common_keywords):
    words = jd_text.split()
    highlighted_text = " ".join([colored(word, "green", attrs=["bold"]) if word.lower() in common_keywords else word for word in words])
    return highlighted_text

# Example Usage
resume_path = 'resume.pdf'
resume_text = extract_text_from_pdf(resume_path)
jd_text = input("Enter job description: ")

match_percentage, common_keywords = calculate_match(resume_text, jd_text)

print(f"\nResume matches the job description by {match_percentage:.2f}%\n")
print("Highlighted Job Description with Matched Keywords:\n")
print(highlight_matched_words(jd_text, common_keywords))
