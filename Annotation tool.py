import streamlit as st
import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from streamlit_player import st_player
import zipfile

# Initialize app
st.title("Multimodal Sarcasm Annotation Tool")

# Sidebar for uploads
st.sidebar.header("Upload Files")

# Context uploads
st.subheader("Context Uploads")
context_video = st.sidebar.file_uploader("Upload Context Video", type=["mp4", "avi", "mov"])
context_audio = st.sidebar.file_uploader("Upload Context Audio", type=["wav", "mp3"])
context_frames_zip = st.sidebar.file_uploader("Upload Context Frames (.zip)", type=["zip"])
context_splenshape = st.sidebar.file_uploader("Upload Context Splenshape (Spectrogram)", type=["png", "jpg"])

# Utterance uploads
st.subheader("Utterance Uploads")
utterance_video = st.sidebar.file_uploader("Upload Utterance Video", type=["mp4", "avi", "mov"])
utterance_audio = st.sidebar.file_uploader("Upload Utterance Audio", type=["wav", "mp3"])
utterance_frames_zip = st.sidebar.file_uploader("Upload Utterance Frames (.zip)", type=["zip"])
utterance_splenshape = st.sidebar.file_uploader("Upload Utterance Splenshape (Spectrogram)", type=["png", "jpg"])

# Extract and display context and utterance data
if context_video and context_audio and context_frames_zip and context_splenshape \
        and utterance_video and utterance_audio and utterance_frames_zip and utterance_splenshape:
    
    # Context Video
    st.subheader("Context Video Preview:")
    st.video(context_video)

    # Context Audio
    y, sr = librosa.load(context_audio)
    st.subheader("Context Audio Waveform:")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Context Spectrogram
    st.subheader("Context Splenshape (Spectrogram):")
    st.image(context_splenshape)

    # Extract Context Frames
    with zipfile.ZipFile(context_frames_zip, 'r') as z:
        z.extractall("context_frames/")
    context_frames = sorted(Path("context_frames").glob("*.png"))
    st.subheader("Context Frames:")
    for i, frame in enumerate(context_frames):
        st.image(str(frame), caption=f"Context Frame {i + 1}")

    # Utterance Video
    st.subheader("Utterance Video Preview:")
    st.video(utterance_video)

    # Utterance Audio
    y, sr = librosa.load(utterance_audio)
    st.subheader("Utterance Audio Waveform:")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Utterance Spectrogram
    st.subheader("Utterance Splenshape (Spectrogram):")
    st.image(utterance_splenshape)

    # Extract Utterance Frames
    with zipfile.ZipFile(utterance_frames_zip, 'r') as z:
        z.extractall("utterance_frames/")
    utterance_frames = sorted(Path("utterance_frames").glob("*.png"))
    st.subheader("Utterance Frames:")
    for i, frame in enumerate(utterance_frames):
        st.image(str(frame), caption=f"Utterance Frame {i + 1}")

    # Save Annotations
    if st.button("Save Annotations"):
        annotations = {
            "context": {
                "frames": [str(f) for f in context_frames]
            },
            "utterance": {
                "frames": [str(f) for f in utterance_frames]
            }
        }
        with open("annotations.json", "w") as f:
            json.dump(annotations, f, indent=4)
        st.success("Annotations saved to annotations.json!")
