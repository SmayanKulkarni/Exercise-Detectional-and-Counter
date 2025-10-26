import os
import numpy as np
import pickle
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization, 
                                     Conv1D, MaxPooling1D)
# Re-importing to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# --- Configuration ---
PROCESSED_DIR = os.path.expanduser('~/Downloads/Exercises/processed_data')
SEQ_LENGTH = 60
NUM_BODY_LANDMARKS = 22  # 33 total - 11 face
# ---------------------

# --- 1. Load Data ---
sequences = []
labels = []

exercise_folders = sorted([f for f in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, f))])
NUM_CLASSES = len(exercise_folders)

if NUM_CLASSES == 0:
    print(f"Error: No processed data found in {PROCESSED_DIR}")
    exit()

print(f"Loading data from {NUM_CLASSES} classes: {exercise_folders}")

for exercise_label in exercise_folders:
    exercise_path = os.path.join(PROCESSED_DIR, exercise_label)
    for npy_file in os.listdir(exercise_path):
        if npy_file.endswith('.npy'):
            data = np.load(os.path.join(exercise_path, npy_file))
            if data.shape == (SEQ_LENGTH, NUM_BODY_LANDMARKS, 3):
                data_flattened = data.reshape(SEQ_LENGTH, -1) # Will be (60, 66)
                sequences.append(data_flattened)
                labels.append(exercise_label)
            else:
                print(f"Skipping file with incorrect shape: {npy_file} has {data.shape}")

print(f"Loaded {len(sequences)} sequences.")

# --- 2. Encode Labels ---
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
# Using to_categorical for one-hot encoding
labels_categorical = to_categorical(labels_encoded, num_classes=NUM_CLASSES)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print(f"Label encoder classes: {le.classes_}")

# --- 3. Prepare Data ---
X = np.array(sequences, dtype='float32')
# Using the one-hot encoded labels
y = labels_categorical
print(f"X shape: {X.shape}, y shape: {y.shape}") # y shape will be (samples, num_classes)

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4452, stratify=y)

# --- 5. Build the Conv1D + LSTM Model ---
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', 
                 input_shape=(SEQ_LENGTH, X.shape[2]))) # X.shape[2] is 66
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.4))
model.add(LSTM(32, return_sequences=True, activation='relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(NUM_CLASSES, activation='softmax')) # Output layer is still softmax

# Switched loss and metrics back
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) # 'accuracy' is standard here
model.summary()

# --- 6. Train the Model (with Callbacks) ---
print("\n--- Starting Model Training ---")

# --- THIS IS THE CORRECTED LINE ---
# Now monitoring 'val_accuracy' with mode 'max'
early_stopping = EarlyStopping(monitor='val_accuracy', 
                             patience=10, 
                             mode='max',  # 'max' because higher accuracy is better
                             restore_best_weights=True)

log_dir = os.path.join(
    "logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=(2, 4))
print(f"TensorBoard log directory: {log_dir}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, tensorboard_callback]
)

print("\n--- Training Complete ---")

# --- 7. Evaluate and Save ---
print("Evaluating model with best weights (from val_accuracy):")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nBest Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

model.save('exercise_classifier_conv1d_lstm.h5')
print("Model saved as 'exercise_classifier_conv1d_lstm.h5'")