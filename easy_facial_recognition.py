# Code Anis - Defend Intelligence
import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import argparse
from pathlib import Path
import os
import ntpath
import mediapipe as mp

parser = argparse.ArgumentParser(description='Easy Facial Recognition App')
parser.add_argument('-i', '--input', type=str, required=True, help='directory of input known faces')

print('[INFO] Starting System...')
print('[INFO] Importing pretrained model..')
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("pretrained_model/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print('[INFO] Importing pretrained model..')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

heart_img = cv2.imread("heart.png", cv2.IMREAD_UNCHANGED)
heart_img = cv2.resize(heart_img, (80, 80))

ok_img = cv2.imread("ok.png", cv2.IMREAD_UNCHANGED)
ok_img = cv2.resize(ok_img, (80, 80))

def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


def encode_face(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_detector(image, 1)
    face_encodings_list = []
    landmarks_list = []

    for face_location in face_locations:
        shape = pose_predictor_68_point(image, face_location)
        try:
            descriptor = face_encoder.compute_face_descriptor(image, shape, num_jitters=1)
            face_encodings_list.append(np.array(descriptor))
        except Exception as e:
            print(f"[ERROR] Face encoding failed: {e}")
            continue

        shape_np = face_utils.shape_to_np(shape)
        landmarks_list.append(shape_np)

    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list


def easy_face_reco(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = [vector <= tolerance for vector in vectors]
        name = known_face_names[result.index(True)] if True in result else "Unknown"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)


def is_heart_pose(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return distance < 0.06

def is_ok_pose(hand_landmarks):
    # İşaret parmağı yukarıda, başparmak kıvrılmışsa 'ok' işareti
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    thumb_tip = hand_landmarks.landmark[4]
    wrist = hand_landmarks.landmark[0]

    # Basit bir tanım: işaret parmağı diğerlerinden yukarıda ve başparmak bileğe yakın
    is_index_up = index_tip.y < middle_tip.y and index_tip.y < thumb_tip.y
    is_thumb_down = abs(thumb_tip.y - wrist.y) < 0.05

    return is_index_up and is_thumb_down

if __name__ == '__main__':
    args = parser.parse_args()

    print('[INFO] Importing faces...')
    face_to_encode_path = Path(args.input)
    files = list(face_to_encode_path.rglob('*.jpg')) + list(face_to_encode_path.rglob('*.png'))
    if len(files) == 0:
        raise ValueError(f'No faces detected in the directory: {face_to_encode_path}')

    known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]
    known_face_encodings = []
    for file_ in files:
        image = PIL.Image.open(file_)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)

    print('[INFO] Faces well imported')
    print('[INFO] Starting Webcam...')
    video_capture = cv2.VideoCapture(0)
    print('[INFO] Webcam well started')
    print('[INFO] Detecting...')

    while True:
        ret, frame = video_capture.read()
        easy_face_reco(frame, known_face_encodings, known_face_names)

        # MediaPipe ellerle kalp kontrolü
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            hand_poses = [hand for hand in result.multi_hand_landmarks]

            if len(hand_poses) == 2:
                poses = [is_heart_pose(hand) for hand in hand_poses]
                if all(poses):
                    h, w, _ = frame.shape
                    x, y = w // 2 - 40, h // 2 - 40
                    for c in range(0, 3):
                        frame[y:y+80, x:x+80, c] = (
                            heart_img[:, :, c] * (heart_img[:, :, 3] / 255.0) +
                            frame[y:y+80, x:x+80, c] * (1.0 - heart_img[:, :, 3] / 255.0)
                        )
                    cv2.putText(frame, "❤️ Kalp!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            elif len(hand_poses) == 1:
                if is_ok_pose(hand_poses[0]):
                    h, w, _ = frame.shape
                    x, y = w // 2 - 40, h // 2 - 40
                    for c in range(0, 3):
                        frame[y:y+80, x:x+80, c] = (
                            ok_img[:, :, c] * (ok_img[:, :, 3] / 255.0) +
                            frame[y:y+80, x:x+80, c] * (1.0 - ok_img[:, :, 3] / 255.0)
                        )
                    cv2.putText(frame, "👌 Tamam!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.imshow('Easy Facial Recognition App', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('[INFO] Stopping System')
    video_capture.release()
    cv2.destroyAllWindows()
    hands.close()
