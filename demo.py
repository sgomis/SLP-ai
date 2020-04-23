import librosa
import numpy as np
import azure.cognitiveservices.speech as speechsdk
from keras.models import load_model

speech_key, service_region = "38e0202520e54a16af74fdcc0aede3bf", "eastus"  # this needs to be hidden


# Emotion detection
# Dictionary assigning emotion labels to integers
emotions = {"0": "neutral", "1": "calm", "2": "happy", "3": "sad",
            "4": "angry", "5": "fearful", "6": "disgust", "7": "surprised"}

# Load pre-trained model (trained on using Azure ML Studio and RAVDESS dataset)
speechtoemotion = load_model('/Users/aravind/Desktop/MS Challenge/Demo/SpeechToEmotion_Model.h5')
# speechtoemotion.summary()

# Acquire input through microphone
child_input = '/Users/aravind/Desktop/MS Challenge/Demo/demo1.wav'

# Featurize input into mfccs
X, samplerate = librosa.load(child_input, res_type='kaiser_fast')
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=samplerate, n_mfcc=40).T, axis=0)
demo1_input = np.expand_dims(np.expand_dims(mfccs, axis=2), axis=0)

# Predict emotion using model
demo1_class = speechtoemotion.predict_classes(demo1_input)
emotion = emotions[str(demo1_class[0])]

# Emotion output
print('Predicted Emotion: ', emotion)

# Speech to Text using Azure Speech API
# Creates a recognizer with the given settings
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

# Replace test.wav with audio file
audio_config = speechsdk.audio.AudioConfig(filename=child_input)
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

# Creates a speech recognizer using a file as audio input, language set to US English
speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config, language="en-US", audio_config=audio_config)

result = speech_recognizer.recognize_once()
print('Predicted text: ', result.text)
