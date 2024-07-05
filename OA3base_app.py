import streamlit as st
import joblib
import pandas as pd
from PyPDF2 import PdfReader
import docx
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from docx import Document

# Ensure you have the necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the vectorizer and models (replace with your own model loading code)
vectorizer = joblib.load("tfidfvect.pkl")
best_log_reg = joblib.load("Logistic_regression.pkl")
best_svm = joblib.load("SVM.pkl")
best_knn = joblib.load("KNN.pkl")
best_naive_bayes = joblib.load("Naive_Bayes.pkl")
best_grad_boost = joblib.load("Gradient_Boosting.pkl")
voting_clf = joblib.load("Voting_Classifier.pkl")

# Stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Contact details of co-founders
co_founders = {
    "Ndivho Mamphiswana": "mamphiswanan@gmail.com",
    "Masixole Nondumo": "m.nondumo@gmail.com",
    "Carol Tshabane": "ctshabane@gmail.com",
    "Muwanwa Tshikovhi": "tshikovhimuwanwa@gmail.com",
    "Mookodi Mokoatle": "lungelo618@gmail.com",
    "Simcelile Zinya": "simceh@gmail.com"
}

def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert text to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatization
    text = re.sub(r'\b\w\b', '', text)  # Remove single characters
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    return text

def preprocess_text(text):
    cleaned_text = clean_text(text)
    vect_text = vectorizer.transform([cleaned_text])
    return vect_text

def plot_wordcloud(text):
    wordcloud = WordCloud(stopwords=stop_words, background_color='white', max_words=100, max_font_size=50, scale=3).generate(text)
    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()
    return fig

def extract_text_from_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "text/csv":
        df = pd.read_csv(file)
        return " ".join(df.astype(str).values.flatten())
    elif file.type == "application/pdf":
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        return ""

def main():
    st.title("News Classifier: Empowering Smarter Reading")
    st.markdown("Automatically categorize news articles into Business, Technology, Sports, and more.")

    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    if selection == "Information":
        st.info("General Information")

        with st.expander("How to Use the App"):
            st.markdown("""
            1. Go to the **Prediction** section.
            2. Choose whether to **Enter text** or **Upload a file**.
            3. If entering text, type or paste your news article into the text area and click **Next**.
            4. If uploading a file, select your file and click **Next**.
            5. Choose your preferred machine learning model and click **Next**.
            6. Click **Category** to classify the text and view the results.
            """)

        with st.expander("Why Use This App?"):
            st.markdown("""
            - **Efficiency:** Quickly categorize articles without manual effort.
            - **Accuracy:** Utilizes advanced machine learning models for precise categorization.
            - **Insightful:** Gain insights into trending topics and article categories.
            - **User-friendly:** Easy-to-use interface suitable for both readers and editors.
            """)

        with st.expander("Contact Details of Co-Founders"):
            for founder, email in co_founders.items():
                st.write(f"**{founder}:** {email}")

    elif selection == "Prediction":
        st.info("Prediction with ML Models")

        if 'step' not in st.session_state:
            st.session_state.step = 1

        if st.session_state.step == 1:
            input_method = st.radio("Choose input method:", ("Enter text", "Upload file"))

            if input_method == "Enter text":
                news_text = st.text_area("Enter Text", "Type Here")
                if st.button("Next", key="step1_next_text"):
                    st.session_state.news_text = news_text
                    st.session_state.step = 2

            elif input_method == "Upload file":
                uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "pdf", "docx"])
                if uploaded_file is not None:
                    news_text = extract_text_from_file(uploaded_file)
                    if st.button("Next", key="step1_next_file"):
                        st.session_state.news_text = news_text
                        st.session_state.step = 2

        elif st.session_state.step == 2:
            predictor = st.selectbox("Choose Model", ["Logistic Regression", "SVM", "KNN", "Naive Bayes", "Gradient Boosting", "Voting Classifier"], key="step2_select")
            if st.button("Next", key="step2_next"):
                st.session_state.predictor = predictor
                st.session_state.step = 3

        elif st.session_state.step == 3:
            vect_text = preprocess_text(st.session_state.news_text)

            if st.session_state.predictor == "Logistic Regression":
                model = best_log_reg
            elif st.session_state.predictor == "SVM":
                model = best_svm
            elif st.session_state.predictor == "KNN":
                model = best_knn
            elif st.session_state.predictor == "Naive Bayes":
                model = best_naive_bayes
            elif st.session_state.predictor == "Gradient Boosting":
                model = best_grad_boost
            else:
                model = voting_clf

            if st.button("Category", key="step3_category"):
                prediction = model.predict(vect_text)
                category = prediction[0]
                probabilities = model.predict_proba(vect_text)[0]

                st.success(f"Text Categorized as: **{category}**")

                prob_df = pd.DataFrame(probabilities, index=model.classes_, columns=['Probability'])
                st.subheader("Prediction Probabilities")
                st.write(prob_df)

                st.subheader("Word Cloud")
                wordcloud_fig = plot_wordcloud(st.session_state.news_text)
                st.pyplot(wordcloud_fig)

                st.session_state.step = 1

if __name__ == '__main__':
    main()
