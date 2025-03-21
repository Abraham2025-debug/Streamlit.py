import streamlit as st
import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import zipfile
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips
from streamlit_player import st_player
import speech_recognition as sr

# Initialize app
st.title("Multimodal Sarcasm Annotation Tool")

# Sidebar for uploads
st.sidebar.header("Upload Files")
context_video = st.sidebar.file_uploader("Upload Context Video", type=["mp4", "avi", "mov"], key="context_video")
utterance_video = st.sidebar.file_uploader("Upload Utterance Video", type=["mp4", "avi", "mov"], key="utterance_video")
frame_folder = st.sidebar.file_uploader("Upload Extracted Frames (Folder as .zip)", type=["zip"])

if context_video and utterance_video and frame_folder:
    # Load and concatenate videos
    context_clip = VideoFileClip(context_video.name)
    utterance_clip = VideoFileClip(utterance_video.name)
    full_clip = concatenate_videoclips([context_clip, utterance_clip])

    # Display concatenated video
    st.subheader("Combined Video Preview (Context + Utterance):")
    st.video(full_clip.write_videofile("combined_video.mp4", codec="libx264"))

    # Extract and play audio
    audio_path = "extracted_audio.wav"
    full_clip.audio.write_audiofile(audio_path)

    # Audio visualization
    y, sr_rate = librosa.load(audio_path)
    st.subheader("Audio Waveform:")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr_rate, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Transcribe audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
            st.subheader("Transcript:")
            st.text(transcript)
        except sr.UnknownValueError:
            st.warning("Could not transcribe audio.")

    # Extract frames from the zip folder
    with zipfile.ZipFile(frame_folder, 'r') as z:
        z.extractall("frames/")

    frame_files = sorted(list(Path("frames").glob("*.png")))
    st.subheader("Extracted Frames:")

    frame_annotations = {}
    for i, frame in enumerate(frame_files):
        st.image(str(frame), caption=f"Frame {i + 1}")
        face_indices = st.text_input(f"Face indices for Frame {i + 1}", key=f"face_{i}")
        facial_cue = st.text_input(f"Facial cue type for Frame {i + 1}", key=f"cue_{i}")
        sarcasm_label = st.selectbox(
            f"Annotate Frame {i + 1}", ["Select Label", "Sarcastic", "Non-Sarcastic"], key=f"sarcasm_{i}")

        frame_annotations[str(frame)] = {
            "face_indices": face_indices,
            "facial_cue": facial_cue,
            "sarcasm_label": sarcasm_label
        }

    if st.button("Save Annotations"):
        annotations = {
            "transcript": transcript,
            "frame_annotations": frame_annotations
        }
        with open("annotations.json", "w") as f:
            json.dump(annotations, f, indent=4)
        st.success("Annotations saved to annotations.json!")
