import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import preprocess_input

# === Configuration ===
dataset_dir = r'C:\Users\Sagi Ashwitha\Downloads\medicinal_plants\Indian Medicinal Leaves Image Datasets\Medicinal plant dataset'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# === Data Augmentation & Preprocessing ===
datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === Save Class Names ===
class_names = list(train_gen.class_indices.keys())
os.makedirs("model", exist_ok=True)
with open("model/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# === Build Model ===
base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
base_model.trainable = True  # Fine-tuning!

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train Model ===
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# === Save Model ===
model.save("model/xception_plant_model.h5")
print("✅ Model trained and saved to 'model/xception_plant_model.h5'")
