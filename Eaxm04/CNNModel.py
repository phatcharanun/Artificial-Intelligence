import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
import os

# กำหนด path
base_dir = 'Eaxm04/train'

# 1. โหลดข้อมูลและตรวจสอบ
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# ตรวจสอบจำนวนคลาส
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Normalization
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 2. สร้างโมเดล
model = Sequential([
    Input(shape=(128, 128, 3)),  # ปรับตามข้อมูลจริง
    Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2), padding='valid'),
    Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2), padding='valid'),
    Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2), padding='valid'),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # ปรับตามจำนวนคลาสจริง
])

# 3. Compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# แสดงสรุปโมเดล
model.summary()

# 4. Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 5. Save model
model.save('my_model.h5')
print("Training completed and model saved!")

