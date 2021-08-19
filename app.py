from flask import Flask, request, render_template

# tensor flow imports
from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession

# keras imports
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# general imports
import numpy as np
import wikipedia
from os import listdir

# GUI related code to compress total usage
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# initalise app
app = Flask(__name__)

# load the trained model
model_path = "Model/model_inception.h5"
model = load_model(model_path)

# plants directories
dirs = [d for d in listdir('TL/data/train')]


def predict_image(path):
  img = image.load_img(path, target_size=(224, 224))
  image_arr = image.img_to_array(img)
  image_arr = image_arr / 255  # normalising image
  image_arr = np.expand_dims(image_arr, axis=0)
  prediction = model.predict(image_arr)
  prediction = np.argmax(prediction, axis=1)
  result = dirs[prediction[0]]
  return wikipedia.summary(result, sentences=2)


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
  if request.method == "POST":
    file = request.files['file']
    file_path = 'Uploads/' + file.filename
    file.save(file_path)
    prediction = predict_image(file_path)
    return prediction

  return None


if __name__ == "__main__":
  app.run(port=5000, debug=True)
