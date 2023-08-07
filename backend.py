from flask import Flask, render_template, request, jsonify
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from PIL import Image
from keras.preprocessing import image

app = Flask(__name__, template_folder='templates')

# Data Preparation
# Set the path to your original dataset containing images in each class folder
original_dataset_dir = 'C:\\Users\\varsh\\OneDrive\\Desktop\\project\\1.Rice_Dataset_Original'

# Set the base directory where you want to save the divided data
base_dir = 'C:\\Users\\varsh\\OneDrive\\Desktop\\project\\new data'
os.makedirs(base_dir, exist_ok=True)

# List of class names
class_names = ['bacterial_leaf_blight', 'Brown_spot', 'Healthy', 'Hispa', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'Shath Blight', 'Tungro']

# Create train, validation, and test directories inside the base directory
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

val_dir = os.path.join(base_dir, 'validation')
os.makedirs(val_dir, exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# Split the data into train, validation, and test sets (adjust the test_size and validation_size as needed)
for class_name in class_names:
    class_dir = os.path.join(original_dataset_dir, class_name)
    images = os.listdir(class_dir)

    # Split the data into train and remaining data
    train_images, remaining_images = train_test_split(images, test_size=0.2, random_state=42)

    # Further split the remaining data into validation and test sets
    val_images, test_images = train_test_split(remaining_images, test_size=0.5, random_state=42)

    # Create class directories inside train, validation, and test directories
    train_class_dir = os.path.join(train_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)

    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(val_class_dir, exist_ok=True)

    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(test_class_dir, exist_ok=True)

    # Move images to their respective directories
    for train_image in train_images:
        src_path = os.path.join(class_dir, train_image)
        dest_path = os.path.join(train_class_dir, train_image)
        shutil.copy(src_path, dest_path)

    for val_image in val_images:
        src_path = os.path.join(class_dir, val_image)
        dest_path = os.path.join(val_class_dir, val_image)
        shutil.copy(src_path, dest_path)

    for test_image in test_images:
        src_path = os.path.join(class_dir, test_image)
        dest_path = os.path.join(test_class_dir, test_image)
        shutil.copy(src_path, dest_path)


# Machine Learning Model Training
# Set the paths to your training and validation data directories
train_dir = 'C:\\Users\\varsh\\OneDrive\\Desktop\\project\\new data\\train'
validation_dir = 'C:\\Users\\varsh\\OneDrive\\Desktop\\project\\new data\\validation'

# Define data augmentation for training data and normalization for validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Set the batch size
batch_size = 16
# Set the image size (adjust according to your model input requirements)
image_size = (224, 224)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the ResNet50V2 pre-trained model
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom dense layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(9, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)
# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False
# Compile the model with a higher initial learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# Define callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
# Train the model with increased epochs and augmented data
epochs = 50  
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Function to preprocess image for model input
def preprocess_image(file):
    # Load the image using PIL
    img = Image.open(file)

    # Resize the image to the target size (e.g., 224x224) expected by the model
    img = img.resize((224, 224))

    # Convert the image to an array
    img_array = image.img_to_array(img)

    # Expand dimensions to create a batch of size 1 (required for the model)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image (e.g., normalize pixel values)
    preprocessed_image = tf.keras.applications.resnet_v2.preprocess_input(img_array)

    return preprocessed_image

# Function to predict disease based on the preprocessed image
def predict_disease(image_file):
    # Preprocess the image (resize and normalize) for the model input
    preprocessed_image = preprocess_image(image_file)

    # Use the machine learning model for disease prediction
    prediction = model.predict(preprocessed_image)

    # Assuming prediction is a one-hot encoded array representing probabilities for each class,
    # find the class with the highest probability as the predicted disease.
    predicted_class_index = tf.argmax(prediction, axis=-1).numpy()

    # Get the class name based on the index
    predicted_disease = class_names[predicted_class_index[0]]

    # Get the probability of the predicted class
    probability = prediction[0][predicted_class_index][0]

    return predicted_disease, probability
# Route to serve the index.html (home) page
@app.route('/')
def home():
    return render_template('index.html')

# Route to serve the about.html page
@app.route('/about')
def about():
    return render_template('about.html')

# Route to serve the upload.html page
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    # Perform disease prediction using the uploaded image
    predicted_disease, probability = predict_disease(image_file)
    # Convert the numpy float32 data to regular Python float data
    probability = probability.item()
    
    # Return the response as a JSON object
    return jsonify({'result': predicted_disease, 'probability': probability}), 200

    

if __name__ == '__main__':
    app.run(debug=True)