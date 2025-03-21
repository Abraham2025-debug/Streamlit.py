import streamlit as st
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
import ffmpeg
import os
import tempfile
from streamlit_player import st_player
import speech_recognition as sr

def extract_audio(video_path, audio_path):
    try:
        ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
    except ffmpeg.Error as e:
        st.error("Error extracting audio from video: " + str(e))

def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    st.pyplot(plt)

def main():
    st.title("Multimodal Annotation Tool")

    context_video = st.file_uploader("Upload Context Video", type=["mp4", "avi", "mpeg4"], key="context")
    utterance_video = st.file_uploader("Upload Utterance Video", type=["mp4", "avi", "mpeg4"], key="utterance")

    if context_video and utterance_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as context_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as utterance_temp:
            context_temp.write(context_video.read())
            utterance_temp.write(utterance_video.read())

            combined_audio_path = tempfile.mktemp(suffix=".wav")
            extract_audio(context_temp.name, combined_audio_path)

            st_player(context_temp.name)
            st_player(utterance_temp.name)

            st.write("### Extracted Audio Waveform:")
            plot_waveform(combined_audio_path)

if __name__ == "__main__":
    main()
