# use a streamlit to create webste 
import streamlit as st 
# add audio recorder streamlit to add recording audio 
from audio_recorder_streamlit import audio_recorder
# add whisper google to generate the AI to transfer speech to text from Audio record and upload file as well 
import whisper
import time
import tempfile
from transformers import pipeline
import torch
torch.set_num_threads(1)





# add layout of the web page 
st.set_page_config(page_title="Sentiment Analysis", page_icon="🙂" , layout="wide")

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

# add streamlit container 
with st.container():
    st.markdown('<div class="toolbar">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("Image/logo.png", width=120)
    with col2:
        st.markdown("<h1>AI Voice & Text Sentiment Analysis: Understanding What People Feel</h1>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
with st.container():
     st.markdown('<div class="toolbar">', unsafe_allow_html=True)


st.subheader("")

with st.container(): 
    column1, spacer , column2 = st.columns([2, 0.1 , 2])
    with column1: 
        # add upload file first where user can upload file to generate text from video 
        Upload_file = st.file_uploader("Please upload your file here", type=["txt", "csv", "jpg", "png", "mp3", "mp4"])
        if Upload_file is not None: 
         st.write(Upload_file.name)
         st.write(Upload_file.type)
         st.write(Upload_file.size, "bytes")

        if Upload_file is not None: 
          temp_uploadfile =  f"temp_uploaded_audio.{Upload_file.name.split('.')[-1]}"
          with open(temp_uploadfile, "wb") as f: 
           f.write(Upload_file.read())  
           
        if st.button("Start Transcription", key="UploadButton"):   
         st.info("Transcription start ...... Please wait")
        
        # ProgressText = st.empty()
        # ProgressBar = st.progress(0)
        # for i in range (50): 
          #  time.sleep(0.5)
          #  ProgressText.text(f"🔊 Transcribing audio... {i * 100 // 50}%")
          #  ProgressBar.progress(i * 100 // 50)
            
            
            
         model = whisper.load_model("base")
         
         if Upload_file: 
           Result = model.transcribe(temp_uploadfile)
           st.success("Transcription Successfully Completed")
           st.text_area("📝 Transcription  Result ", Result["text"], height=300)
           
           st.info("🎧 Detecting tone and emotion from voice...")
           emotion = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device = "cpu")
           emotion_result = emotion(temp_uploadfile)
           st.success("✅ Emotion analysis completed!")
           
           emotion_labels = {
           "sad": "Sad",
           "ang": "Angry",
           "neu": "Neutral",
            "hap": "Happy"
            }
           st.write("### 🎭 Detected Emotions:")

           for e in emotion_result:
            full_label = emotion_labels.get(e['label'], e['label'])
            st.write(f"**{full_label}**: {e['score']:.2f}")
           
         
         else: 
             st.error("⚠️ Error not file upload, Please upload file before staring Transcription") 
          
    with spacer: 
        
         st.markdown(
        """<div style="border-left:1px solid gray;height:300px"></div>""",
           unsafe_allow_html=True,
            )       
    with column2: 
       # add record allow to user to record Audio then get a text from it.    
       Recorder = audio_recorder(text="Record your Audio here", recording_color="#FF0000", neutral_color="#6E6E6E")
       if Recorder: 
        st.audio(Recorder, format="audio/wav")
        temp_file = "recorded_audio.wav"
        with open(temp_file, "wb") as f:
         f.write(Recorder)
       
       if st.button("Start Transcription", key="Record_Button"):    
         st.info("Transcription start ...... Please wait")
       
       # ProgressText = st.empty()
       #  ProgressBar = st.progress(0) 
       
       # for i in range (30): 
       #    time.sleep(0.5) 
       #    ProgressText.text(f"🔊 Transcribing audio... {i * 100 // 30}%")
       #    ProgressBar.progress(i * 100 // 30) 
            

         model = whisper.load_model("base")

      
         if Recorder:   
           Result = model.transcribe(temp_file)
           st.success("Transcription Successfully Completed")
           st.text_area("📝 Transcription  Result ", Result["text"], height=300)
                     
           
           st.info("🎧 Detecting tone and emotion from voice...")
           emotion1 = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device="cpu")
           emotion1_result = emotion1(temp_file)
           st.success("✅ Emotion analysis completed!")
           
           st.write("### 🎭 Detected Emotions:")
           emotion_labels = {
           "sad": "Sad",
           "ang": "Angry",
           "neu": "Neutral",
            "hap": "Happy"
            }
           for e in emotion1_result:
            full_label = emotion_labels.get(e['label'], e['label'])
            st.write(f"**{full_label}**: {e['score']:.2f}")
           
         
         else: 
            st.error("⚠️ Error not Audio Record, Please record your audio before staring Transcription ") 
            
   
   

