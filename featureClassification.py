# 1. Euclidean Distance
# 2. Cosine Similarity
# 3. SVM
# 4. KNN
# 5. ANN
import numpy as np
from featureExtraction import feature_extraction_vgg_face
import os
import time

feature_ds_path = r'storage\model\feature_ds\feature_ds.npz'


def euclidean_dist_classify(face_pixels):
    t1_start = time.process_time()
    if os.path.exists(feature_ds_path):
        feature_ds = np.load(feature_ds_path)
    euclid_dist_list = []
    audit_feature = feature_extraction_vgg_face(face_pixels)
    for feature in feature_ds['feature']:
        euclidean_dist = np.linalg.norm(audit_feature - feature)
        euclid_dist_list.append(euclidean_dist)
    min_distance = np.min(euclid_dist_list)
    min_index = euclid_dist_list.index(min_distance)
    label = feature_ds['label'][min_index]
    print('Label: %s, distance: %.3f' % (label, min_distance))
    t1_stop = time.process_time()
    print("Recognize face time: " + str(t1_stop-t1_start))
    return label, min_distance


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x, axis = 0, keepdims = True))
#     return e_x / e_x.sum(axis=0)


# def euclidean_dist_classify(face_pixels):
#     if os.path.exists(feature_ds_path):
#         feature_ds = np.load(feature_ds_path)
#     euclid_dist_list = []
#     audit_feature = feature_extraction_vgg_face(face_pixels)
#     for feature in feature_ds['feature']:
#         euclidean_dist = np.linalg.norm(audit_feature - feature)
#         euclid_dist_list.append(euclidean_dist)
#     probability_list = softmax(euclid_dist_list)
#     print(np.max(probability_list))
#     min_distance = np.min(euclid_dist_list)
#     min_index = euclid_dist_list.index(min_distance)
#     label = feature_ds['label'][min_index]
#     print('Label: %s (%.3f), distance: %.3f' % (label, probability_list[min_index],min_distance))
#     return label, probability_list[min_index]


def cosine_similarity_classify(face_pixels):
    t1_start = time.process_time()
    if os.path.exists(feature_ds_path):
        feature_ds = np.load(feature_ds_path)
    probability_list = []
    audit_feature = feature_extraction_vgg_face(face_pixels)
    for feature in feature_ds['feature']:
        probability = np.dot(audit_feature, feature)/(np.linalg.norm(audit_feature)*np.linalg.norm(feature))
        probability_list.append(probability)
    max_prob = np.max(probability_list)
    max_index = probability_list.index(max_prob)
    label = feature_ds['label'][max_index]
    print('Label: %s (%.3f)' % (label, max_prob))
    t1_stop = time.process_time()
    print("Recognize face time: " + str(t1_stop-t1_start))
    return label, max_prob*100


from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


def train_svm_classification_model():
    if os.path.exists(feature_ds_path):
        feature_ds = np.load(feature_ds_path)
    feature_list = feature_ds['feature']
    name_list = feature_ds['label']
    le = LabelEncoder()
    labels = le.fit_transform(name_list)
    np.save(r'storage\model\feature_classify\svm_classes.npy', le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(feature_list, labels, test_size=0.2, random_state=233)
    print('Dataset: train=%d, test=%d' % (X_train.shape[0], X_test.shape[0]))
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)
    score_train = accuracy_score(y_train, yhat_train)
    score_test = accuracy_score(y_test, yhat_test)
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
    save_path = r'storage\model\feature_classify\svm_classification_model.sav'
    pickle.dump(model, open(save_path, 'wb'))
    print('Model saved')


def svm_classify(face_pixels):
    t1_start = time.process_time()
    audit_feature = feature_extraction_vgg_face(face_pixels)
    save_path = r'storage\model\feature_classify\svm_classification_model.sav'
    svm_classify_model = pickle.load(open(save_path, 'rb'))
    samples = np.expand_dims(audit_feature, axis=0)
    yhat_class = svm_classify_model.predict(samples)
    yhat_prob = svm_classify_model.predict_proba(samples)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    labels = LabelEncoder()
    labels.classes_ = np.load(r'storage\model\feature_classify\svm_classes.npy')
    predict_names = labels.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    t1_stop = time.process_time()
    print("Recognize face time: " + str(t1_stop-t1_start))
    return predict_names[0], class_probability


from sklearn.neighbors import KNeighborsClassifier


def train_knn_classification_model():
    if os.path.exists(feature_ds_path):
        feature_ds = np.load(feature_ds_path)
    feature_list = feature_ds['feature']
    name_list = feature_ds['label']
    le = LabelEncoder()
    labels = le.fit_transform(name_list)
    np.save(r'storage\model\feature_classify\knn_classes.npy', le.classes_)
    X_train, X_test, y_train, y_test = train_test_split(feature_list, labels, test_size=0.2, random_state=233)
    print('Dataset: train=%d, test=%d' % (X_train.shape[0], X_test.shape[0]))
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(feature_list,labels)
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)
    score_train = accuracy_score(y_train, yhat_train)
    score_test = accuracy_score(y_test, yhat_test)
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
    save_path = r'storage\model\feature_classify\knn_classification_model.sav'
    pickle.dump(model, open(save_path, 'wb'))
    print('Model saved')


def knn_classify(face_pixels):
    t1_start = time.process_time()
    audit_feature = feature_extraction_vgg_face(face_pixels)
    save_path = r'storage\model\feature_classify\knn_classification_model.sav'
    svm_classify_model = pickle.load(open(save_path, 'rb'))
    samples = np.expand_dims(audit_feature, axis=0)
    yhat_class = svm_classify_model.predict(samples)
    class_index = yhat_class[0]
    labels = LabelEncoder()
    labels.classes_ = np.load(r'storage\model\feature_classify\knn_classes.npy')
    predict_names = labels.inverse_transform(yhat_class)
    print('Predicted: %s' % (predict_names[0]))
    t1_stop = time.process_time()
    print("Recognize face time: " + str(t1_stop-t1_start))
    return predict_names[0]

