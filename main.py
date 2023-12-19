import librosa
import soundfile
import os, glob, pickle
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split


model = pickle.load(open('emotion_classification-model.pkl', 'rb'))


#Setting Title of App
st.title("Emotion Recognition Prediction")

#Uploading the dog image
audio_file = st.file_uploader("Choose an audio file...", type=["wav", "mp4", "mp3", "m4a"])
submit = st.button('Predict')
#On predict button click
if submit:


    if audio_file is not None:
        def extract_feature(file_name, mfcc, chroma, mel):
            # Open the sound file using the soundfile library
            with soundfile.SoundFile(file_name) as sound_file:
                # Read the sound file data and sample rate
                X = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate

                # Compute the short-time Fourier transform (stft) of the audio signal
                stft = np.abs(librosa.stft(X))

                # Initialize an empty array to store the extracted features
                result = np.array([])

                # If the 'mfcc' flag is True, extract Mel-frequency cepstral coefficients (MFCC)
                if mfcc:
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                    result = np.hstack((result, mfccs))

                # If the 'chroma' flag is True, extract chroma feature
                if chroma:
                    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                    result = np.hstack((result, chroma))

                # If the 'mel' flag is True, extract mel spectrogram feature
                if mel:
                    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                    result = np.hstack((result, mel))

            # Return the concatenated array of extracted features
            return result


        def transform_data(file):
            x = []
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            return x



        def handler(file_zap):
            data = transform_data(file_zap)
            print(np.array([data]).reshape(1, -1).shape)
            res = model.predict(np.array([data]).reshape(1, -1))
            return st.title(res)

        handler(audio_file)









