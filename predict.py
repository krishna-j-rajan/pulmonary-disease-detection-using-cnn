from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import normalize
import numpy as np
import sys
if len(sys.argv)==1:
    print("Missing file path argument")
    exit()

IMG_SIZE = 224
model = keras.models.load_model("./pneumonia_train.h5")
# path = './ZhangLabData/CellData/chest_xray/test/PNEUMONIA/BACTERIA-1351146-0001.jpeg'
path = sys.argv[1]
img = keras.preprocessing.image.load_img(
    path, target_size=(IMG_SIZE, IMG_SIZE)
)
img_array = np.array(img).reshape(-1, IMG_SIZE,IMG_SIZE, 3)
img_array = normalize(img_array, axis = 1)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
class_names = ['Normal', 'Bacteria-Pneumonia', 'Virus-Pneumonia']

print( 
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)





