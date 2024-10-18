import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numpy import arange

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)
        
        # Using st.columns for layout
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(indices[0]):
                col.image(filenames[indices[0][i]])
            else:
                col.text("") # Placeholder if there are fewer images than columns
                
    else:
        st.header("Some error occurred in file upload")

# Load the training and validation loss dictionaries
train_loss = {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1}
val_loss = {1: 0.6, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.2}

# Generate random accuracy values for demonstration
train_accuracy = {1: 0.8, 2: 0.85, 3: 0.9, 4: 0.92, 5: 0.95}
val_accuracy = {1: 0.7, 2: 0.75, 3: 0.8, 4: 0.85, 5: 0.88}

# Retrieve each dictionary's values
train_loss_values = list(train_loss.values())
val_loss_values = list(val_loss.values())
train_accuracy_values = list(train_accuracy.values())
val_accuracy_values = list(val_accuracy.values())

# Generate a sequence of integers to represent the epoch numbers
epochs = list(train_loss.keys())

# Generate random precision and recall values for demonstration
train_precision = {1: 0.75, 2: 0.78, 3: 0.81, 4: 0.85, 5: 0.88}
val_precision = {1: 0.65, 2: 0.68, 3: 0.72, 4: 0.75, 5: 0.78}

train_recall = {1: 0.70, 2: 0.72, 3: 0.75, 4: 0.78, 5: 0.80}
val_recall = {1: 0.60, 2: 0.62, 3: 0.65, 4: 0.68, 5: 0.70}

# Retrieve each dictionary's values
train_precision_values = list(train_precision.values())
val_precision_values = list(val_precision.values())
train_recall_values = list(train_recall.values())
val_recall_values = list(val_recall.values())

# Plot and label the training and validation loss values
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(epochs, train_loss_values, label='Training Loss')
ax1.plot(epochs, val_loss_values, label='Validation Loss')

# Add in a title and axes labels
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')

# Set the tick locations
ax1.set_xticks(arange(0, len(epochs) + 1, 2))

# Display the plot
ax1.legend(loc='best')
st.pyplot(fig1)

# Plot and label the training and validation accuracy values
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(epochs, train_accuracy_values, label='Training Accuracy')
ax2.plot(epochs, val_accuracy_values, label='Validation Accuracy')

# Add in a title and axes labels
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')

# Set the tick locations
ax2.set_xticks(arange(0, len(epochs) + 1, 2))

# Display the plot
ax2.legend(loc='best')
st.pyplot(fig2)

# Plot and label the training and validation precision values
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(epochs, train_precision_values, label='Training Precision')
ax3.plot(epochs, val_precision_values, label='Validation Precision')

# Add in a title and axes labels
ax3.set_title('Training and Validation Precision')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Precision')

# Set the tick locations
ax3.set_xticks(arange(0, len(epochs) + 1, 2))

# Display the plot
ax3.legend(loc='best')
st.pyplot(fig3)

# Plot and label the training and validation recall values
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(epochs, train_recall_values, label='Training Recall')
ax4.plot(epochs, val_recall_values, label='Validation Recall')

# Add in a title and axes labels
ax4.set_title('Training and Validation Recall')
ax4.set_xlabel('Epochs')
ax4.set_ylabel('Recall')

# Set the tick locations
ax4.set_xticks(arange(0, len(epochs) + 1, 2))

# Display the plot
ax4.legend(loc='best')
st.pyplot(fig4)
