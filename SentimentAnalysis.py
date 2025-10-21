# use streamlit tom creae a website 
import streamlit as st
# use audion recorder for recording the audio
from audio_recorder_streamlit import audio_recorder
# use whiper to trascrip the from audio to vedio 
import whisper
# use transformers and import pipline for using to create emotion and felling 
from transformers import pipeline
# create temfile to stor the file and be eble allow to work in stream lit cloud 
import tempfile
import torch
import os
import time


torch.set_num_threads(1)


st.set_page_config(page_title="Sentiment Analysis", page_icon="🙂", layout="wide")

st.markdown("""
<style>
.toolbar {
    background-color: #2C3E50;
    padding: 10px;
    border-radius: 8px;
}
.toolbar h2 {
    color: white;
    text-align: center;
    margin: 0;
    line-height: 50px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("tiny")

@st.cache_resource
def load_emotion_model():
    return pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device="cpu")

model = load_whisper_model()
emotion_model = load_emotion_model()

# place the emotion that can displayed in the streamlit cloud 
EMOTION_LABELS = {
    "sad": "Sad 😢",
    "ang": "Angry 😠",
    "neu": "Neutral 😐",
    "hap": "Happy 😊"
}



def save_temp_file(data, suffix):
    """Save uploaded or recorded data to a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name

def transcribe_and_analyze(audio_path):
    """Run Whisper transcription and emotion detection."""
    try:
        result = model.transcribe(audio_path)
        st.success("✅ Transcription completed successfully!")
        st.text_area("📝 Transcription Result", result["text"], height=300)

        st.info("🎧 Detecting tone and emotion from voice...")
        emotion_result = emotion_model(audio_path)
        st.success("✅ Emotion analysis completed!")

        st.write("### 🎭 Detected Emotions:")
        for e in emotion_result:
            label = EMOTION_LABELS.get(e["label"], e["label"])
            st.write(f"**{label}**: {e['score']:.2f}")
    except FileNotFoundError:
        st.error("⚠️ FFmpeg not found. Add 'ffmpeg' to packages.txt.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


with st.container():
    st.markdown('<div class="toolbar">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("Image/logo.png", width=120)
    with col2:
        st.markdown("<h1>AI Voice & Text Sentiment Analysis: Understanding What People Feel</h1>", unsafe_allow_html=True)
    st.markdown('<div class="toolbar">', unsafe_allow_html=True)

st.subheader("")


with st.container():
    col_upload, spacer, col_record = st.columns([2, 0.1, 2])


    with col_upload:
        st.header("📤 Upload an Audio File")
        uploaded_file = st.file_uploader(
            "Upload file (mp3, wav, mp4, etc.)",
            type=["mp3", "wav", "mp4", "m4a"]
        )

        if uploaded_file:
            st.write(f"**File Name:** {uploaded_file.name}")
            st.write(f"**Type:** {uploaded_file.type}")
            st.write(f"**Size:** {uploaded_file.size} bytes")

            temp_path = save_temp_file(uploaded_file.read(), suffix=f".{uploaded_file.name.split('.')[-1]}")

            if st.button("🎙️ Start Transcription for Uploaded File"):
                st.info("Transcription in progress... please wait ⏳")
                transcribe_and_analyze(temp_path)
        else:
            st.write("Upload an audio file to begin.")

 
    with spacer:
        st.markdown(
            """<div style="border-left:1px solid gray;height:400px"></div>""",
            unsafe_allow_html=True,
        )


    with col_record:
        st.header("🎤 Record Your Audio")
        recorder_audio = audio_recorder(text="Click to record your voice", recording_color="#FF0000", neutral_color="#6E6E6E")

        if recorder_audio:
            st.audio(recorder_audio, format="audio/wav")
            temp_file = save_temp_file(recorder_audio, suffix=".wav")

            if st.button("🎙️ Start Transcription for Recorded Audio"):
                st.info("Transcription in progress... please wait ⏳")
                transcribe_and_analyze(temp_file)
        else:
            st.write("Press the record button to capture your voice.")
