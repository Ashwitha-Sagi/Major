import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

dataset_dir = r'C:\Users\Sagi Ashwitha\Downloads\medicinal_plants\Indian Medicinal Leaves Image Datasets\Medicinal plant dataset'

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

datagen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)

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

class_names = list(train_gen.class_indices.keys())
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

os.makedirs("model", exist_ok=True)
model.save("model/xception_plant_model.h5")
print("✅ Model trained and saved!")
