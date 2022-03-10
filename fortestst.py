# def feature_extractor(face_aligned):
#     landmark = pose_predictor(face_aligned, dlib.rectangle(0, 0, face_aligned.shape[0], face_aligned.shape[1]))
#     face_descriptor = face_encoder.compute_face_descriptor(face_aligned, landmark, num_jitters=0)
#     return face_descriptor

# def faceReg(image):
#     faces_aligned, faces_location = face_detector_aligned(image)
#     if (faces_aligned == []):
#         # print('No face in image')
#         pass
#     else:
#         # print('{} face in image'.format(len(faces_aligned)))
#         for i, face_aligned in enumerate(faces_aligned):
#             (x, y, w, h) = rect_to_bb(faces_location[i])
#             cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
#             face_encoded = feature_extractor(face_aligned)
#             distances = list(np.linalg.norm(faces_descriptor - face_encoded, axis=1))
#             min_idx = np.argmin(distances)
#             name, distance = names[min_idx], distances[min_idx]
#             if distance > 0.4:
#                 text = "Unknown face"             
#             else:
#                 text = name
#             cv2.putText(image, '{}_{:.2f}'.format(text, distance), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     return image

# face_encoder = dlib.face_recognition_model_v1(r'storage\model\dlib_face_recognition_resnet_model_v1.dat')
# pose_predictor = dlib.shape_predictor(r'storage\model\shape_predictor_68_face_landmarks.dat')

# def face_detector_aligned(image):  
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces_location = hog_face_detector(image, 0)
#     faces_aligned = []
#     for face_location in faces_location:
#         fa = FaceAligner(pose_predictor)
#         face_aligned = fa.align(image, gray, face_location)
#         face_aligned = cv2.resize(face_aligned, (224, 224), interpolation = cv2.INTER_AREA)
#         faces_aligned.append(face_aligned)
#     return faces_aligned, faces_location

# m = create_model((512,512,3), 2)
# m.load_weights('xxx.h5') # note that weights can be loaded from a full save, not only from save_weights file
# m.save('xxx_3.5.

# import cv2