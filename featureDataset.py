import json
from dataPreparation import data_train_enriched
import os
import cv2
from featureExtraction import feature_extraction_vgg_face
import numpy as np


imageTrain_dir = r'storage\imageTrain'
feature_ds_path = r'storage\model\feature_ds\feature_ds.npz'

def create_feature_ds():
    data_train_enriched()
    if os.path.exists(feature_ds_path):
        os.remove(feature_ds_path)
    feature_ds = []
    label_ds = []
    names, fld_list = read_user_json()
    for i, fld_name in enumerate(os.listdir(imageTrain_dir)):
        sub_dir = os.path.join(imageTrain_dir,fld_name)
        label = names[fld_list.index(fld_name)]
        for img_name in os.listdir(sub_dir):
            img_dir = os.path.join(sub_dir,img_name)
            img = cv2.imread(img_dir)
            embedding = feature_extraction_vgg_face(img)
            feature_ds.append(embedding)
            label_ds.append(label)
    np.savez(feature_ds_path, feature=feature_ds, label=label_ds)


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

