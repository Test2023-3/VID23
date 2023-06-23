import os
import contextlib
import time
import textwrap
import csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import firebase_admin
from firebase_admin import credentials, storage
import whisper
import streamlit as st
import bardapi as bard_module
from nltk.corpus import stopwords

from collections import defaultdict
import heapq
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
import random
from random import shuffle
import re
import json
import tempfile
from pydub import AudioSegment
import streamlit.components.v1 as components
import deep_translator
from deep_translator import GoogleTranslator

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
# Check if the 'punkt' data is already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:  
    nltk.download("punkt")


def mark_button_action(b_key: str):
    st.session_state[b_key] = not st.session_state[b_key]

# Add your Bard token here
os.environ['_BARD_API_KEY'] = "Wwj86guYc9unyPQDkyNlWsR7IGBlXKAc-Gk7s7RfVb9YrEVptkKXYG7Ykg-OcLCYnxtIVA."


if not firebase_admin._apps:
    cred = credentials.Certificate("firebase.json")
    firebase_app = firebase_admin.initialize_app(cred, {
        "storageBucket": "chatgpt-28e16.appspot.com",
    })

def get_available_mp4s(bucket):
    blobs = bucket.list_blobs(prefix="mp3_files/")
    mp4_files = [os.path.basename(blob.name).replace('.mp4', '') for blob in blobs]
    return mp4_files

def extract_audio(video_filepath, audio_filepath):
    video = AudioSegment.from_file(video_filepath, "mp4")
    video.export(audio_filepath, format="mp3")

def transcribe_audio(audio_file_path):
    model = whisper.load_model("large")
    result = model.transcribe(audio_file_path)
    transcript = result["text"]
    return transcript

@contextlib.contextmanager
def get_bard_answer(input_text):
    bard = bard_module.Bard(language="en")
    response = bard.get_answer(input_text)
    content = response['content']
    yield content
    del bard

def get_relevant_context(transcription, user_question):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    all_passages = sent_detector.tokenize(transcription)

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user_question] + all_passages)

    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Get indices of the top 2 most relevant sentences
    top_indices = cosine_similarities.argsort()[:-3:-1]
    relevant_passages = [all_passages[i] for i in top_indices]

    max_context_length = 100  # Set this to the desired character limit
    combined_passages = ''
    for passage in relevant_passages:
        combined_passages += ' ' + passage
        if len(combined_passages.strip()) >= max_context_length:
            break

    context = user_question + ". " + combined_passages.strip()
    return context[:512]

def export_data(data, filename):
    with open(filename, "w", encoding="utf-8") as txtfile:
        for key, value in data.items():
            txtfile.write(f"{key}:\n{value}\n\n")
        st.success(f"Data saved to {filename}")

def translate_text(text):
    translation = GoogleTranslator(source='auto', target='ar').translate(text)
    return translation

def preprocess_text(text):
    stop_words = stopwords.words('english')

    # Tokenize, remove stopwords, and lowercase
    words = simple_preprocess(text)
    words = [word for word in words if word not in stop_words]

    return [words]

def create_dictionary_and_corpus(text):
    preprocessed_text = preprocess_text(text)
    dictionary = corpora.Dictionary(preprocessed_text)
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_text]
    return dictionary, corpus

def train_lda_model(dictionary, corpus, num_topics=5):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics)
    return lda_model

def extract_topic_keywords(lda_model, num_keywords=5):
    topic_keywords = []

    for topic in lda_model.print_topics():
        topic_words = ''.join([char if char.isalnum() or char == ' ' else '' for char in topic[1]])
        keywords = topic_words.split()[:num_keywords]
        topic_keywords.append(keywords)

    return topic_keywords

def generate_concept_map(text):
    dictionary, corpus = create_dictionary_and_corpus(text)
    lda_model = train_lda_model(dictionary, corpus)
    topic_keywords = extract_topic_keywords(lda_model)

    graph = nx.Graph()

    for idx, keywords in enumerate(topic_keywords):
        topic_name = f"Topic {idx}"
        graph.add_node(topic_name)
        for keyword in keywords:
            graph.add_node(keyword)
            graph.add_edge(topic_name, keyword)

    pyvis_graph = Network(notebook=False)
    pyvis_graph.from_nx(graph)

    pyvis_graph.save_graph("output/concept_map.html")
    return "output/concept_map.html"

def initialize_button_states():
    # Initialize Session State Attributes if required
    if not st.session_state.get('submit_question_clicked'):
        st.session_state.submit_question_clicked = False

    if not st.session_state.get('regenerate_answer_clicked'):
        st.session_state.regenerate_answer_clicked = False


st.markdown("<h2 style='text-align: center; color: white; background-color: darkgreen;'>Educational Video Assistant</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black; background-color: lightgray;'>This AI Video Assistant will help you get answers to your questions on educational videos</p>",
            unsafe_allow_html=True)

st.write("""
    <style>
    .arabic-text {
        display: block;
        unicode-bidi: bidi-override;
        direction: rtl;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize translation_transcription, translation_bard_answer, and translation_further_info
if 'translation_transcription' not in st.session_state:
    st.session_state.translation_transcription = ""

if 'translation_bard_answer' not in st.session_state:
    st.session_state.translation_bard_answer = ""

if 'translation_further_info' not in st.session_state:
    st.session_state.translation_further_info = ""

# Initialize st.session_state.transcription
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""

# Transcription Module
st.subheader("Transcription Module")
st.markdown("Please select a video file to transcribe, then translate to Arabic.")
bucket = storage.bucket("chatgpt-28e16.appspot.com")

mp4_list = get_available_mp4s(bucket)
video_name = st.selectbox("Select an mp4", mp4_list)

if 'downloaded_video' not in st.session_state:
    st.session_state.downloaded_video = ""

if st.button("Submit to download video"):
    video_blob = bucket.get_blob(f"mp3_files/{video_name}.mp4")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
        video_filepath = tf.name
        video_blob.download_to_filename(video_filepath)
        st.session_state.downloaded_video = video_filepath

if st.session_state.downloaded_video:
    st.video(st.session_state.downloaded_video)

if st.button("Submit to transcribe"):
    if st.session_state.downloaded_video:
        audio_filepath = "temp.mp3"
        start_time = time.time()
        extract_audio(st.session_state.downloaded_video, audio_filepath)
        transcription_time = time.time() - start_time

        start_time = time.time()
        st.session_state.transcription = transcribe_audio(audio_filepath)
        if os.path.exists(audio_filepath):  # Check if the file exists before removing
            os.remove(audio_filepath)

def extract_key_concepts(text, n=5):
    # Use a regular expression to filter out punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    word_pos_tags = nltk.pos_tag(words)

    word_frequencies = defaultdict(int)
    important_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}  # nouns and adjectives
    stop_words = set(stopwords.words("english"))

    for word, pos_tag in word_pos_tags:
        if word.lower() not in stop_words and pos_tag in important_pos_tags:
            word_frequencies[word] += 1

    most_frequent_words = heapq.nlargest(n, word_frequencies, key=word_frequencies.get)
    concepts = {word: random.uniform(0.3, 1) for word in most_frequent_words}

    return concepts

transcription = st.session_state.transcription
concepts = extract_key_concepts(transcription)

def highlight_concepts(transcription, concepts):
    highlighted_text = transcription
    for concept in concepts:
        color = f"rgba(255, 0, 0, {concepts[concept]})"
        highlight_tag = f'<span style="background-color: {color}">{concept}</span>'
        highlighted_text = highlighted_text.replace(concept, highlight_tag)
    return highlighted_text

def display_highlighted_transcription(transcription, concepts):
    highlighted_text = highlight_concepts(transcription, concepts)
    st.markdown(f"<div>{highlighted_text}</div>", unsafe_allow_html=True)

if st.session_state.transcription:
    display_highlighted_transcription(transcription, concepts)

if 'translate_transcription_arabic_clicked' not in st.session_state:
    st.session_state.translate_transcription_arabic_clicked = False

if st.button("Translate Transcription to Arabic"):
    mark_button_action('translate_transcription_arabic_clicked')

if st.session_state.translate_transcription_arabic_clicked:
    st.session_state.translation_transcription = translate_text(transcription)
    st.markdown(f"<div class='arabic-text'><label>Transcription in Arabic</label><br><textarea style='width: 100%; height: 150px;'>{st.session_state.translation_transcription}</textarea></div>", unsafe_allow_html=True)

if 'bard_answer' not in st.session_state:
    st.session_state.bard_answer = ""
# Initialize Arabic translations of Engage and Reinforcement
if 'translation_bard_answer' not in st.session_state:
    st.session_state.translation_bard_answer = ""

if 'translation_further_info' not in st.session_state:
    st.session_state.translation_further_info = ""

# Engage Module
if st.session_state.transcription:
    st.subheader("Engagement Module")
    st.markdown("Enter a question to get an answer from the video. You can re-generate the answer as many times as you want. You can translate the answer to Arabic as well")
    user_question = st.text_input("Enter a question here to get an answer")

    col1, col2 = st.columns([1, 1])

    if not st.session_state.get('submit_question_clicked'): st.session_state.submit_question_clicked = False
    # Call initialize_button_states after defining the col1 and col2
    initialize_button_states()

    if col1.button("Submit Question"):
        mark_button_action('submit_question_clicked')

    if col2.button("Regenerate Answer"):
        mark_button_action('regenerate_answer_clicked')

    if st.session_state.submit_question_clicked or st.session_state.regenerate_answer_clicked:
        context = get_relevant_context(st.session_state.transcription, user_question)
        answer = context.split(". ", 1)[1]
        with get_bard_answer(context[:512]) as bard_answer:
            st.session_state.bard_answer = bard_answer
        st.text_area("Answer", value=answer, height=150)

        col3, col4 = st.columns([1, 1])

        if not st.session_state.get('translate_answer_arabic_clicked'): st.session_state.translate_answer_arabic_clicked = False
        if col3.button("Translate Answer to Arabic"):
            mark_button_action('translate_answer_arabic_clicked')

        if st.session_state.translate_answer_arabic_clicked:
            st.session_state.translation_bard_answer = translate_text(st.session_state.bard_answer)
            st.markdown(f"<div class='arabic-text'><label>Answer in Arabic</label><br><textarea style='width: 100%; height: 150px;'>{st.session_state.translation_bard_answer}</textarea></div>", unsafe_allow_html=True)
    # Reinforcement Module
    if st.session_state.bard_answer:
        st.subheader("Reinforcement Module")
        st.markdown("Click the Further Information button to get more information about the previous answer you got. You can re-generate the answer as many times as you want. You can translate the answer to Arabic as well. You can create a visual concept map from it")

        col5, col6 = st.columns([1, 1])

        if not st.session_state.get('further_information_clicked'): st.session_state.further_information_clicked = False
        if col5.button("Further Information"):
            mark_button_action('further_information_clicked')

        if not st.session_state.get('regenerate_further_information_clicked'): st.session_state.regenerate_further_information_clicked = False
        if col6.button("Regenerate Further Information"):
            mark_button_action('regenerate_further_information_clicked')

        if st.session_state.further_information_clicked or st.session_state.regenerate_further_information_clicked:
            context = "Generate further information about this: " + st.session_state.bard_answer
            with get_bard_answer(context[:512]) as bard_answer:
                st.session_state.further_info = bard_answer
            st.text_area("Further Information", value=st.session_state.further_info, height=150)

        col7, col8 = st.columns([1, 1])

        if not st.session_state.get('translate_further_info_arabic_clicked'): st.session_state.translate_further_info_arabic_clicked = False
        if col7.button("Translate Further Information to Arabic"):
            mark_button_action('translate_further_info_arabic_clicked')

        if st.session_state.translate_further_info_arabic_clicked:
            st.session_state.translation_further_info = translate_text(st.session_state.further_info)
            st.markdown("<div class='arabic-text'><label>Further Information in Arabic</label></div>", unsafe_allow_html=True)

            st.write("""
                <style>
                .translation-container {
                    width: 100%;
                    height: 150px;
                    text-align: right;
                    direction: rtl;
                    overflow: scroll;
                    border: 1px solid #ccc;
                    padding: 5px 10px;
                    box-sizing: border-box;
                }
                </style>
                """, unsafe_allow_html=True)

            st.markdown(f"<div class='translation-container'>{st.session_state.translation_further_info}</div>", unsafe_allow_html=True)

# Concept Map Module
    if st.session_state.transcription:
        st.subheader("Concept Map")
        if st.button("Generate Concept Map"):
            concept_map_path = generate_concept_map(st.session_state.transcription)
            st.success("Concept map generated!")

        if 'concept_map_path' in locals():
            with open(concept_map_path, "r") as file:
                st.components.v1.html(file.read(), height=600)
            st.download_button("Download Concept Map", concept_map_path, "concept_map.html")
# Save data as text
        if st.button("Export"):
            data = {
                "transcription": st.session_state.transcription,
                "translation_transcription": st.session_state.translation_transcription,
                "user_question": user_question,
                "bard_answer": st.session_state.bard_answer,
                "translation_bard_answer": st.session_state.translation_bard_answer,
                "further_info": st.session_state.further_info,
                "translation_further_info": st.session_state.translation_further_info
            }

            export_data(data, "output_data.txt")
