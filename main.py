import os
import streamlit as st
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from io import BytesIO
from gtts import gTTS
import re
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face API Configuration
hf_api_key = st.secrets["HF_API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {hf_api_key}"}


# Function to Summarize Text Using Hugging Face API
def summarize_text(text, max_length=130, min_length=30):
    payload = {
        "inputs": text,
        "parameters": {"max_length": max_length, "min_length": min_length, "do_sample": False}
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    else:
        return None

# Function to Validate and Extract YouTube Video ID
def extract_video_id(link):
    youtube_regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(youtube_regex, link)
    if match:
        return match.group(1)
    else:
        st.error("Invalid YouTube link. Please provide a valid URL.")
        return None

# Extract Transcript from YouTube Video
def extract_transcript(video_id):
    try:
        with st.spinner("Fetching transcript... This may take a moment."):
            time.sleep(1)  # Delay for 1 second before the request
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            # transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([segment["text"] for segment in transcript_data])
            return transcript
    except Exception as e:
        st.warning("⚠️ This video does not have subtitles. Try another video.")
        st.error(f"Error fetching transcript: {e}")
        return None

# Generate Summary Using Hugging Face API
def generate_summary(transcript_text, summary_ratio, chunk_size=500, overlap=50):
    try:
        total_text_length = len(transcript_text)
        summary_length = max(int(total_text_length * (summary_ratio / 100)), 50)  # Minimum length safeguard
        
        # Splitting transcript into overlapping chunks
        transcript_chunks = []
        start = 0
        while start < total_text_length:
            end = start + chunk_size
            transcript_chunks.append(transcript_text[start:end])
            start += chunk_size - overlap

        progress = st.progress(0)
        chunk_summaries = []
        total_chunks = len(transcript_chunks)
        
        # Adjust max summary length for each chunk
        max_chunk_summary_length = max(summary_length // total_chunks, 50)

        with st.spinner("Generating summary... This may take some time."):
            for i, chunk in enumerate(transcript_chunks):
                summary = summarize_text(chunk, max_length=max_chunk_summary_length, min_length=20)
                if summary:
                    chunk_summaries.append(summary)
                progress.progress((i + 1) / total_chunks)

        # Improved bullet formatting using regex to avoid breaking abbreviations
        sentences = re.split(r'(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', " ".join(chunk_summaries))
        bullet_summary = "\n\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence])

        return bullet_summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

# Convert Text to Speech
def text_to_speech(text, language='en'):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# Main Summarization Process
def summarization():
    st.caption("Enter the YouTube video link and select the summary percentage to generate a bullet-point summary.")
    youtube_link = st.text_input("Enter the YouTube video link here:")
    
    if youtube_link:
        video_id = extract_video_id(youtube_link)
        if video_id:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", width=400)
    
    summary_ratio = st.slider("Select Summary Percentage", min_value=10, max_value=100, value=10, step=10)
    
    if st.button("Summarize"):
        if video_id:
            transcript_text = extract_transcript(video_id)
            if transcript_text:
                st.text("Summarizing the transcript...")
                summary = generate_summary(transcript_text, summary_ratio)
                if summary:
                    st.subheader("Summary:")
                    st.markdown(summary)
                    audio_data = text_to_speech(summary)
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")
                else:
                    st.error("Failed to generate the summary. Please try again.")
        else:
            st.error("Invalid YouTube video link. Please check and try again.")

# Main Function
def main():
    st.title("YouTube Summarizer")
    st.write("Instantly summarize lengthy YouTube videos in bullet points.")
    st.divider()
    summarization()

if __name__ == "__main__":
    main()
