import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="Waste classification.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = {0: 'glass', 1: 'metal', 2: 'paper', 3: 'plastic'}

def predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0

    if img_array.shape[-1] == 4: 
        img_array = img_array[:, :, :3]  

    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = labels[predicted_index]

    return predicted_label
    
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    live=True,
    title="Waste Classification Model",
    description="Upload a photo of the waste and we will classify it."
)

# تشغيل الواجهة
interface.launch(show_error=True)
