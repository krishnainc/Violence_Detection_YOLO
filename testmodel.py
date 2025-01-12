from tensorflow.keras.applications import MobileNetV2

try:
    model = MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
    print("MobileNetV2 loaded successfully!")
except Exception as e:
    print(f"Error loading MobileNetV2: {e}")
