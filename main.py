import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

# Load your data
data = pd.read_csv("environment_data.csv")

# Features
X = data[['temperature', 'co2', 'humidity', 'wind_speed']].values

# Target
y = data['disaster_type'].astype('category').cat.codes.values  # Convert to integers

# Convert to one-hot encoding for multiclass classification
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

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