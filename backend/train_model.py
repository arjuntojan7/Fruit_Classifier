from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import os

# Paths
train_dir = "../dataset/Train"
val_dir = "../dataset/Test"

# Image generators
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ✅ Build model using Functional API + MobileNetV2
input_shape = (224, 224, 3)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

inputs = Input(shape=input_shape)
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(3, activation='softmax')(x)
model = Model(inputs, outputs)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# ✅ Save model
model_dir = "../model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "fruit_model.keras")
model.save(model_path)
print("✅ Model saved at", model_path)
