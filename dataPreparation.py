from keras import Sequential, layers
import numpy as np
import os
import tensorflow as tf
import PIL
import shutil

imageBase_dir = r'storage\imageBase'
imageTrain_dir = r'storage\imageTrain'


def data_train_enriched():   
    if os.path.exists(imageTrain_dir):
        shutil.rmtree(imageTrain_dir)
    data_augmentation = Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),])
    for label in os.listdir(imageBase_dir):
        sub_dir = os.path.join(imageBase_dir,label)
        for j, image in enumerate(os.listdir(sub_dir)):
            image_dir = os.path.join(sub_dir,image)
            base_image = PIL.Image.open(image_dir, 'r')
            base_image = base_image.resize((224, 224))
            save_dir = os.path.join(imageTrain_dir,'{}'.format(label))
            if not os.path.exists(imageTrain_dir):
                os.mkdir(imageTrain_dir)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            base_image.save(os.path.join(save_dir,'{}_0.jpg'.format(j)))
            for i in range(1, 10):
                augmented_image = data_augmentation(tf.expand_dims(base_image, 0), training=True)
                augmented_image = PIL.Image.fromarray(augmented_image[0].numpy().astype(np.uint8))
                augmented_image.save(os.path.join(save_dir,'{}_{}.jpg'.format(j,i)))