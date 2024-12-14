
import os
import cv2
import numpy as np
from PIL import Image
import pyttsx3
from paddleocr import PaddleOCR
import streamlit as st
from gtts import gTTS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

# Initializing Google Generative AI with the provided API key
GEMINI_API_KEY = 'AIzaSyAdRw5RBVvuch-aYoXa0aOS3NHYOKPrJ1Q'  # Replace with a valid API key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Initializing the Generative AI model with Google Gemini for generating descriptions
llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Initializing the text-to-speech engine for converting text into speech
engine = pyttsx3.init()

# Defining custom CSS to style the Streamlit interface
st.markdown(
    """
    <style>
     .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #4b92db;  /* Soft blue */
        margin-top: -20px;
     }
    .subtitle {
        font-size: 18px;
        color: #888;  /* Lighter gray for contrast */
        text-align: center;
        margin-bottom: 20px;
    }
    .feature-header {
        font-size: 24px;
        color: #f0f0f0;  /* Light gray for headers */
        font-weight: bold;
    }
    body {
        background-color: #121212;  /* Dark background */
        color: #f0f0f0;  /* Light text for readability */
    }
    .stButton>button {
        background-color: #6a77f7;  /* Soft purple for buttons */
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4a65c3;  /* Darker purple when hovered */
    }
    .stTextInput>div>input {
        background-color: #1f1f1f;  /* Dark input fields */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Displaying the main title and description on the Streamlit app
st.markdown('<div class="main-title"> SmartSight AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transforming Lives with AI: Real-Time Scene Understanding, Obstacle Detection, Text Reading, and Voice Guidance!</div>', unsafe_allow_html=True)

# Sidebar Features
st.sidebar.image(
    "TTS.jpg",  # Replace with your image path
    width=250
)

st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
    📌 **Features**
    - 🔍 **Describe Scene**: Getting AI insights about the image, including objects and suggestions.
    - 📝 **Extract Text**: Extracting visible text using OCR.
    - 🔊 **Text-to-Speech**: Converting extracted text to speech.

    💡 **How it helps**:
    Assisting visually impaired users by providing scene descriptions, text extraction, and speech.

    🤖 **Powered by**:
    - **Google Gemini API** for scene understanding.
    - **PaddleOCR** for text extraction.
    - **pyttsx3** for text-to-speech.
    """
)

st.sidebar.text_area("📜 Instructions", "Upload an image to start. Choose a feature to interact with.")

# Function to extract text using PaddleOCR
def extract_text_with_paddleocr(image_data):
    """
    Extract text from the uploaded image using PaddleOCR.
    """
    bytes_data = image_data[0]['data']
    np_img = np.frombuffer(bytes_data, np.uint8)
    cv2_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    ocr_result = ocr.ocr(cv2_image)

    extracted_texts = [line[1][0] for line in ocr_result[0]]
    return " ".join(extracted_texts)

# Function to convert text to speech using gTTS
def text_to_speech_gtts(text):
    """
    Convert text to speech and save as an audio file.
    """
    tts = gTTS(text)
    audio_path = "output.mp3"
    tts.save(audio_path)
    return audio_path

# Adding an image upload section to the interface
st.markdown("<h3 class='feature-header'>Upload an Image 📤</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop or browse an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Feature buttons
st.markdown("<h3 class='feature-header'>⚙️ Features</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

scene_button = col1.button("🔍 Describe Scene")
ocr_button = col2.button("📝 Extract Text")
tts_button = col3.button("🔊 Text-to-Speech")

if uploaded_file:
    image_data = [{"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]

    if ocr_button:
        with st.spinner("Extracting text from the image..."):
            extracted_text = extract_text_with_paddleocr(image_data)
            st.markdown("<h3 class='feature-header'>📝 Extracted Text</h3>", unsafe_allow_html=True)
            st.write(extracted_text)

    if tts_button:
        with st.spinner("Converting text to speech..."):
            extracted_text = extract_text_with_paddleocr(image_data)
            if extracted_text.strip():
                audio_path = text_to_speech_gtts(extracted_text)
                st.success("✅ Text-to-Speech Conversion Completed!")
                st.audio(audio_path, format="audio/mp3")
            else:
                st.warning("No text found to convert.")
