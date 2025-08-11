# Step 1: Install dependencies (if not already installed)
# pip install librosa tensorflow scikit-learn pandas numpy

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------- CONFIG --------
DATA_PATH = "speech_emotion_data"  # folder containing audio files
EMOTIONS_MAP = {
    'happy': 'happy',
    'angry': 'angry',
    'sad': 'sad'
}
# ------------------------

# Step 2: Function to extract MFCC features
def extract_features(file_path, n_mfcc=40, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Step 3: Load dataset and extract features
features = []
labels = []

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_label = None
            fname = file.lower()
            # Example: Detect emotion keyword in filename
            for key in EMOTIONS_MAP.keys():
                if key in fname:
                    emotion_label = EMOTIONS_MAP[key]
                    break
            if emotion_label:
                mfccs = extract_features(os.path.join(root, file))
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(emotion_label)

# Step 4: Prepare data for model
X = np.array(features)
X = X[..., np.newaxis]  # Add channel dimension
le = LabelEncoder()
y = le.fit_transform(labels)
y = to_categorical(y)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 7: Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Step 8: Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")
