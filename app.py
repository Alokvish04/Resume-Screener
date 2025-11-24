import streamlit as st
import pickle
import lzma
import docx
import PyPDF2
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

nltk.download('punkt')
nltk.download('stopwords')

#loading models
# `clf.pkl` is LZMA-compressed; open it with `lzma.open` then `pickle.load`
with lzma.open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
tfidfd = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

#web app
def main():
    st.title("AI Resume Screener")
    uploaded_file = st.file_uploader('Upload Resume', type=['pdf','txt'])
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]


        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales Manager",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PMQ",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        inverse_mapping = {v: k for k, v in category_mapping.items()}

        # Handle both cases: id or job name
        if isinstance(prediction_id, str):
            job_id = inverse_mapping.get(prediction_id, "Unknown")
            job_name = prediction_id
        else:
            job_name = category_mapping.get(prediction_id, "Unknown")
            job_id = prediction_id

        st.write(f"Predicted Category: {job_name} ({job_id})")

#python main
if __name__ == "__main__":
    main()
