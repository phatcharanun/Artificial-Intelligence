import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation

# Create the model
model = Sequential([
    Input(shape=(128, 128, 3)),#ปรับขนาด input ตาม dataset+สีภาพ
    Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2), padding='valid'),
    Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2), padding='valid'),
    Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Activation('relu'),
    Dense(2, activation='softmax')#ปรับจำนวน class ที่ output layer
])

# Compile the model

optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# Display model summary
model.summary()



# 1. Preprocessing & Labeling (ทำตรงนี้)
base_dir = 'Eaxm04/train' # ชื่อโฟลเดอร์หลักที่มีโฟลเดอร์ Mouse และ lip อยู่ข้างใน

train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2, # แบ่งข้อมูล 20% ไว้ตรวจสอบระหว่างเทรน
    subset="training",
    seed=123,
    image_size=(128, 128), # ปรับขนาดภาพ (Resize)
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

# การทำ Normalization (หาร 255)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# 2. Compile Model (กำหนดค่าการเทรน)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# 3. Model Fitting (เริ่มการฝึกสอน)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10 # จำนวนรอบที่เทรน ลองปรับเพิ่มลดได้
)

# 4. บันทึก Model เก็บไว้ใช้
model.save('my_model.h5')
print("เทรนเสร็จและบันทึกโมเดลเรียบร้อยแล้ว!")

