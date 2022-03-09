import cv2
import dlib
from imutils.face_utils import rect_to_bb
# from imutils.face_utils import FaceAligner


hog_face_detector = dlib.get_frontal_face_detector() 

def face_detector_hog(image):
    faces_location_ = hog_face_detector(image, 0)
    faces_img = []
    faces_location = []
    for face_location_ in faces_location_:
        (x, y, w, h) = rect_to_bb(face_location_)
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224), interpolation = cv2.INTER_AREA)
        faces_img.append(face)
        faces_location.append((x, y, w, h))
    return faces_img, faces_location

cnn_face_detector = dlib.cnn_face_detection_model_v1(r'storage\model\mmod_human_face_detector.dat')

def face_detector_cnn(image):
    boxes = cnn_face_detector(image)
    faces_img = []
    faces_location = []
    for box in boxes:
        res_box = process_boxes(box)
        (x, y, w, h) = res_box[0], res_box[1], res_box[2]-res_box[0], res_box[3]-res_box[1]
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224), interpolation = cv2.INTER_AREA)
        faces_img.append(face)
        faces_location.append((x, y, w, h))
    return faces_img, faces_location

def process_boxes(box):
    xmin = box.rect.left()
    ymin = box.rect.top()
    xmax = box.rect.right()
    ymax = box.rect.bottom()
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


from mtcnn import MTCNN

detector = MTCNN()

def face_detector_mtcnn(image):
    results = detector.detect_faces(image)
    faces_img = []
    faces_location = []
    for result in results:
        score = result["confidence"]
        if score > 0.90:
            (x, y, w, h) = result['box']
            i = (h-w)/2
            a = round(x-i)
            b = a+h
            face_img = image[y+2:y+h-2, a+2:b-2]
            faces_img.append(face_img)
            faces_location.append((a,y,h,h))
    return faces_img, faces_location


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

def face_detector_haarcascades(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_location = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_img = []
    for (x,y,w,h) in faces_location:
        face_image = image[y:y+h, x:x+w]
        faces_img.append(face_image)
    return faces_img, faces_location
