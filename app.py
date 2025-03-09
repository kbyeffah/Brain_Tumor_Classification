import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import models
import requests
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env
load_dotenv()

# Retrieve API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Disable oneDNN optimizations for consistent results
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ensure TensorFlow is running in eager mode
tf.config.run_functions_eagerly(True)

# Define class labels for prediction results
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

# Detailed explanations for each class
EXPLANATIONS = {
    "Glioma": (
        "**Glioma** is a type of tumor that occurs in the brain and spinal cord, originating from glial cells. "
        "These tumors can be low-grade (slow-growing) or high-grade (aggressive). \n\n"
        "**Symptoms:** Headaches, seizures, blurred vision, memory loss, personality changes. \n\n"
        "**Treatment:** Surgery, radiation therapy, chemotherapy, targeted therapy.\n\n"
        "**Risk Factors:** Age, genetics, radiation exposure."
    ),
    "Meningioma": (
        "**Meningioma** is usually a benign tumor that grows in the meninges (brain's protective layers). "
        "It often grows slowly and may not require treatment unless symptoms occur.\n\n"
        "**Symptoms:** Headaches, vision problems, weakness, seizures.\n\n"
        "**Treatment:** Observation, surgery, radiation therapy.\n\n"
        "**Risk Factors:** More common in women, linked to hormones, prior radiation exposure."
    ),
    "No Tumor": (
        "**No Tumor Detected!** The scan does not show a tumor. However, persistent symptoms should be evaluated by a doctor.\n\n"
        "**Possible Causes of Symptoms:** Migraines, infections, neurological disorders, stress.\n\n"
        "**Next Steps:** If symptoms continue, consult a neurologist for further evaluation."
    ),
    "Pituitary Tumor": (
        "**Pituitary Tumor** occurs in the pituitary gland, which controls hormone production. "
        "Most are benign but can cause hormonal imbalances.\n\n"
        "**Symptoms:** Vision problems, weight gain/loss, excessive thirst, irregular menstrual cycles.\n\n"
        "**Treatment:** Medications, surgery, radiation therapy.\n\n"
        "**Risk Factors:** Family history, genetic mutations."
    ),
}

def load_vgg_model(model_path):
    """Load the pre-trained model if the file exists."""
    if not os.path.exists(model_path):
        st.error("Model file not found! Please ensure the model file is in the project directory.")
        return None
    return load_model(model_path, compile=False)

# Load the model
MODEL_PATH = "fixed_model.keras"
model = load_vgg_model(MODEL_PATH)

# Streamlit UI
st.title("Brain Tumor Classification")
st.write("Upload a brain MRI image for classification.")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = preprocess_input(img_array_expanded)

    # Display uploaded image
    st.image(img, caption="Uploaded MRI Image", use_container_width=True)

    if model is not None:
        # Perform prediction
        predictions = model.predict(img_array_preprocessed)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = np.max(predictions) * 100

        # Display classification result
        st.subheader("Classification Result")
        st.write(f"Predicted Tumor Type: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Display explanation
        st.subheader("What Does This Mean?")
        st.markdown(EXPLANATIONS[predicted_class])
    else:
        st.error("Model not loaded. Please check the model file.")

# Add a separator before the chatbot
st.markdown("---")



# Function to interact with Groq chatbot
def get_chatbot_response(user_input):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful AI that answers brain tumor-related questions. Keep responses brief and to the point. When asked for definitions or explanations, provide only a short, direct answer without additional details, examples, or classifications unless explicitly requested. "},
            {"role": "user", "content": user_input}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        if response.status_code == 200:
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")
        else:
            return f"Error {response.status_code}: {response_data.get('error', {}).get('message', 'Unknown error')}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"

# Custom CSS for styling the input and send icon
import streamlit as st

st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        border: 1px solid #444;
        border-radius: 20px;
        padding: 10px;
        background-color: #222;
        width: 100%;
        max-width: 400px;
        margin: auto;
    }
    .chat-input {
        height: 45px;
        font-size: 17px; /* Slightly increased */
        border: none;
        width: 100%;
        max-width: 400px;
        border-radius: 10px;
        padding: 10px;
        background-color: #333;
        color: white;
        outline: none;
    }
    .send-button {
        cursor: pointer;
        font-weight: bold;
        text-align: center;
         background-color: #111f3f;
        color: white;
        transition: background 0.3s ease-in-out;
        margin-top: 10px;
        border: none;
        border-radius: 10px;
        width: 100%;
        max-width: 400px;
        height: 45px;
        padding: 12px;
        font-size: 17px; /* Slightly increased */
    }
    .send-button:hover {
        background-color: #888888; /* Slightly darker gray on hover */
    }

    /* Mobile View */
    @media (max-width: 600px) {
        .chat-container {
            width: 90%;
        }
        .chat-input {
            width: 100%;
            font-size: 16px; /* Slightly increased */
            height: 40px;
            padding: 8px;
        }
        .send-button {
            width: 128px; /* 2 inches */
            font-size: 16px; /* Slightly increased */
            height: 40px;
            padding: 8px;
        }
    }
</style>

""", unsafe_allow_html=True)

st.write("Ask me any brain tumor-related questions")

user_input = st.text_input("", placeholder="Enter your question...", key="input_box", label_visibility="collapsed")

st.markdown("""  
    <div style="display: flex; justify-content: center; width: 100%; margin-top: 10px;">
        <button class="send-button" onclick="sendMessage()">Send</button>
    </div>
""", unsafe_allow_html=True)

if user_input:
    st.write("")  # Adds a space before generating the response
    with st.spinner("Thinking..."):
        response = get_chatbot_response(user_input)
    st.write(response)
