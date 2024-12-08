
import os
import cv2
import numpy as np
from PIL import Image
import pyttsx3
import pytesseract  
import streamlit as st
from gtts import gTTS
from paddleocr import PaddleOCR
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

# Setting the path to Tesseract OCR executable for extracting text
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initializing Google Generative AI with the provided API key
GEMINI_API_KEY = 'AIzaSyAdRw5RBVvuch-aYoXa0aOS3NHYOKPrJ1Q'  # Replacing with a valid API key
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

# Function to convert text into speech in a non-blocking way (using a separate thread)
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)  
    engine.runAndWait()  

image = file.open("TTS.jpg")
# Sidebar Features
st.sidebar.image(image,width=250)

# Adding the sidebar with the app description and features for users to explore
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
    - **Tesseract OCR** for text extraction.
    - **pyttsx3** for text-to-speech.
    """
)

# Adding a text box under the sidebar description for instructions
st.sidebar.text_area("📜 Instructions", "Upload an image to start. Choose a feature to interact with:  1 Describe the Scene, 2 Extract Text, 3 Listen to it")

# Defining the function to process the uploaded image and generate a description using Google Generative AI
def extract_text_from_image2(image_data):
    """Generating a scene description using Google Generative AI."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(["""You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
                                        1. Tell the user what text present in image, read it out to understand him. 
                                        """, image_data[0]])
    A = response.text
    return response.text

# Defining the function to extract text from an uploaded image using PaddleOCR
def extract_text_with_paddleocr(image_data):
    """
    Extracting text from an uploaded image using PaddleOCR.

    Parameters:
    image_data (list): A list containing dictionaries with image metadata and byte data.

    Returns:
    str: Extracted text from the image.
    """
    # Extracting bytes from the image data
    bytes_data = image_data[0]['data']

    # Converting bytes to a NumPy array
    np_img = np.frombuffer(bytes_data, np.uint8)

    # Decoding the NumPy array into an OpenCV image
    cv2_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Initializing PaddleOCR for text extraction
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Setting language to English

    # Performing OCR on the OpenCV image
    ocr_result = ocr.ocr(cv2_image)

    # Extracting text from OCR results
    extracted_texts = [line[1][0] for line in ocr_result[0]]
    extracted_text = " ".join(extracted_texts)

    return extracted_text

# Defining the function to convert text to speech using gTTS and save the audio
def text_to_speech_gtts(text):
    """
    Converting the given text to speech using gTTS and saving the audio file.

    Parameters:
    text (str): Text to convert to speech.

    Returns:
    str: Path to the saved audio file.
    """
    tts = gTTS(text)
    audio_path = "output.mp3"
    tts.save(audio_path)
    return audio_path

# Defining the function to generate a scene description from an image using Google Generative AI
def generate_scene_description(input_prompt, image_data):
    """Generating a scene description using Google Generative AI."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

# Function to process and prepare the uploaded image for further processing
def input_image_setup(uploaded_file):
    """Preparing the uploaded image for processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")

# Adding an image upload section to the interface
st.markdown("<h3 class='feature-header'>Upload an Image 📤</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop or browse an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Adding buttons for the available features
st.markdown("<h3 class='feature-header'>⚙️ Features</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

scene_button = col1.button("🔍 Describe Scene")
ocr_button = col2.button("📝 Extract Text")
tts_button = col3.button("🔊 Text-to-Speech")

# Defining the input prompt for generating a scene description
input_prompt = """
You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
1. List of items detected in the image with their purpose.
2. Overall description of the image.
3. Suggestions for actions or precautions for the visually impaired.
"""

import multiprocessing

# Updating the text-to-speech function to save and generate audio files
def speak_text(text):
    import pyttsx3
    engine = pyttsx3.init()
    
    # Saving the speech to an audio file
    audio_filename = "Audioclip.mp3"
    
    # Saving the audio and running it
    engine.save_to_file(text, audio_filename)
    engine.runAndWait()
    engine.stop()

# Processing the user interactions
if uploaded_file:
    
    image_data = input_image_setup(uploaded_file)

    if scene_button:
        with st.spinner("Generating scene description..."):
            response = generate_scene_description(input_prompt, image_data)
            st.markdown("<h3 class='feature-header'>🔍 Scene Description</h3>", unsafe_allow_html=True)
            st.write(response)

    if ocr_button:
        with st.spinner("Extracting text from the image..."):
            text = extract_text_with_paddleocr(image_data)
            st.markdown("<h3 class='feature-header'>📝 Extracted Text</h3>", unsafe_allow_html=True)
            st.write(text)
            
    if tts_button:
        with st.spinner("Converting text to speech..."):
            text = extract_text_with_paddleocr(image_data)
            speak_text(text)
            st.markdown("<h3 class='feature-header'>🔊 Speech Output</h3>", unsafe_allow_html=True)
            st.write(f"Speaking: {text}")
