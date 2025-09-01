import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Convert to TFLite with optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Reduces model size
tflite_model = converter.convert()

# Save the TFLite model
with open('chatbot.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ TFLite model saved as 'chatbot.tflite'")