import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import pandas as pd

def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    precision = tp / (tf.reduce_sum(y_pred) + 1e-8)
    recall = tp / (tf.reduce_sum(y_true) + 1e-8)
    return 2 * (precision * recall) / (precision + recall + 1e-8)

# Load CSV without headers
data = pd.read_csv("data/farm_alert_multiclass_training_data.csv", header=None)

# Replace column names with the first row
data.columns = data.iloc[0]

# Remove the first row from data
data = data[1:].reset_index(drop=True)

# Features (all columns except target)
X = data.drop(columns=['Disaster_Label']).astype(float)

# Target
y = data['Disaster_Label'].astype('category').cat.codes.values

# Convert to one-hot encoding
y = to_categorical(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_features = X_train.shape[1]
num_classes = y_train.shape[1]

model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer for multiclass
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',f1_metric])

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy*100:.2f}%")

# predictions = model.predict(X_test)
# predicted_classes = np.argmax(predictions, axis=1)

# Save the entire model to a file
model.save("natural_disaster_classifier.h5")

model.summary()
print("\n\n")
print(history.history.keys())

# ---- Accuracy Plot ----
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("accuracy.png")


# ---- F1 Score Plot ----
plt.figure(figsize=(8,5))
plt.plot(history.history['f1_metric'], label='Train F1 Score')
plt.plot(history.history['val_f1_metric'], label='Validation F1 Score')
plt.title('Model F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid()
plt.savefig("f1_score.png")

# ---- Loss Plot ----
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("loss.png")