from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import backend as K
from model_ import deep_rank_model
from dataPreparation import data_train_enriched
import os
from keras.optimizers import gradient_descent_v2
import cv2
from featureExtraction import img_normalize
import json
from keras.utils import to_categorical


def read_user_json():
    names = []
    fld_list = []
    with open(r'storage\something\users.json', encoding='utf-8') as json_file:
        data = json.load(json_file)
        users = data['users']
        for user in users:
            name = user['name']
            fld = user['fld_name']
            fld_list.append(fld)
            names.append(name)
    return names, fld_list


imageTrain_dir = r'storage\imageTrain'

def preprocessing():
    names_, fld_list = read_user_json()
    imgs_norm = []
    # labels = []
    names = []
    for fld in os.listdir(imageTrain_dir):
        sub_dir = os.path.join(imageTrain_dir,fld)
        for image in os.listdir(sub_dir):
            img_path = os.path.join(sub_dir,image)
            img = cv2.imread(img_path)
            img_norm = img_normalize(img)
            imgs_norm.append(img_norm)                   
            names.append(names_[fld_list.index(fld)])
    labels = to_categorical(names)
    return imgs_norm, labels, names


batch_size=24
lr=0.001
epochs=2000

_EPSILON = K.epsilon()
def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    loss = 0.
    g = 1.
    for i in range(0, batch_size, 3):
        try:
            q_embedding = y_pred[i]
            p_embedding = y_pred[i+1]
            n_embedding = y_pred[i+2]
            D_q_p = K.sqrt(K.sum((q_embedding - p_embedding)**2))
            D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
            loss = loss + g + D_q_p - D_q_n
        except:
            continue
    loss = loss/batch_size*3
    return K.maximum(loss, 0)


def image_batch_generator(images, labels, batch_size):
    labels = np.array(labels)
    while True:
        batch_paths = np.random.choice(a = len(images), size = batch_size//3)
        input_1 = []
        
        for i in batch_paths:
            pos = np.where(labels == labels[i])[0]
            neg = np.where(labels != labels[i])[0]
            
            j = np.random.choice(pos)
            while j == i:
                j = np.random.choice(pos)
             
            k = np.random.choice(neg)
            while k == i:
                k = np.random.choice(neg)
            
            input_1.append(images[i])
            input_1.append(images[j])
            input_1.append(images[k])

        input_1 = np.array(input_1)
        input = [input_1, input_1, input_1]
        yield(input, np.zeros((batch_size, )))


def train_triplet_loss():
    data_train_enriched()

    X, y, names = preprocessing()  

    filepath = r'storage\model\triplet_weight.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    deep_rank_model_ = deep_rank_model()
    sgd = gradient_descent_v2.SGD(lr=lr)
    deep_rank_model_.compile(loss=_loss_tensor, optimizer=sgd)
    deep_rank_model_.fit_generator(
        generator=image_batch_generator(X, y, batch_size),
        steps_per_epoch=len(X)//batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list
        )



