import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# โหลดโมเดล
model = tf.keras.models.load_model('my_model.h5')

# กำหนดชื่อคลาส
class_names = ['Mouse', 'lip']

# ระบุภาพที่ต้องการทดสอบ
img_path = 'D:/AI/Eaxm04/Input4.jpg'   

# โหลดและ preprocess ภาพ
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
class_index = np.argmax(pred)
confidence = pred[0][class_index]

# แสดงผล
print(f"Input image: {img_path}")
print(f"Prediction: {class_names[class_index]} ({confidence:.2f})")
