import tensorflow as tf
import os

# ใช้ path จากตำแหน่งของไฟล์ปัจจุบัน
# ตรวจสอบ path
base_dir = os.path.dirname(__file__)
test_dir = os.path.join(base_dir, 'test')
print(f"Current file location: {base_dir}")
print(f"Looking for test folder at: {test_dir}")
print(f"Test folder exists: {os.path.exists(test_dir)}")

# ถ้าไม่มีโฟลเดอร์ test
if not os.path.exists(test_dir):
    # แสดงรายการโฟลเดอร์ใน directory ปัจจุบัน
    print("Available folders in current directory:")
    for item in os.listdir(base_dir):
        print(f"  - {item}")

# โหลดโมเดลที่ train เสร็จแล้ว
model = tf.keras.models.load_model('my_model.h5')
# โหลด test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=32
)       

# Normalize
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Evaluate
loss, accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {accuracy:.4f}")
