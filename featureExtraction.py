# Deep Approach (Industrial Giant Nets)
# 1. VGG (VGGFace)
# 2. Face Recognition API
# 3. FaceNet Keras
import model_
from keras import Model
import numpy as np


vgg_face_model = model_.vgg_face_model()
vgg_face_model.load_weights(r'storage\model\vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=vgg_face_model.layers[0].input, outputs=vgg_face_model.layers[-2].output)

def img_normalize(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean = face_pixels.mean()
    std  = face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    return face_pixels

# def img_normalize(face_pixels):
#     face_pixels = face_pixels.astype('float32')
#     face_pixels = face_pixels / 255.
#     return face_pixels

# type(face_pixels) = np.array
def feature_extraction(model, face_pixels):
    face_pixels = img_normalize(face_pixels)
    samples = np.expand_dims(face_pixels,axis=0)
    yhat = model.predict(samples)
    return yhat[0]

def feature_extraction_vgg_face(face_pixels):
    embedding = feature_extraction(vgg_face_descriptor, face_pixels)
    return embedding