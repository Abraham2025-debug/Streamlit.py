import streamlit as st
import cv2
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
import speech_recognition as sr
from streamlit_player import st_player
import tempfile
import shutil
import zipfile
import ffmpeg


def extract_audio(video_path, audio_path):
    video = AudioSegment.from_file(video_path)
    video.export(audio_path, format="wav")


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_paths = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{i}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    cap.release()
    return frame_paths, fps


def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    plt.plot(y)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    st.pyplot(plt)


def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not transcribe"


def main():
    st.title("Multimodal Annotation Tool")
    context_video = st.file_uploader("Upload Context Video", type=["mp4", "avi"])
    utterance_video = st.file_uploader("Upload Utterance Video", type=["mp4", "avi"])

    if context_video and utterance_video:
        context_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        context_temp.write(context_video.read())
        utterance_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        utterance_temp.write(utterance_video.read())

        combined_audio_path = "combined_audio.wav"
        extract_audio(context_temp.name, combined_audio_path)
        plot_waveform(combined_audio_path)

        transcript = transcribe_audio(combined_audio_path)
        st.subheader("Transcript:")
        st.write(transcript)

        output_folder = tempfile.mkdtemp()
        context_frames, _ = extract_frames(context_temp.name, output_folder)
        utterance_frames, _ = extract_frames(utterance_temp.name, output_folder)

        st.subheader("Context Frames:")
        st.image(context_frames, width=200)

        st.subheader("Utterance Frames:")
        st.image(utterance_frames, width=200)

        shutil.rmtree(output_folder)


if __name__ == "__main__":
    main()
