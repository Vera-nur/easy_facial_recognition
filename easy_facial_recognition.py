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
import speech_recognition as sr
import threading
import speech_recognition as sr

keyword_checking = False
keyword_success = False

def threaded_keyword_check(expected_keyword):
    global keyword_checking, keyword_success
    keyword_checking = True
    keyword_success = listen_for_keyword(expected_keyword)
    keyword_checking = False
    

with sr.Microphone() as source:
    print("Mikrofon dinleniyor...")
    recognizer = sr.Recognizer()
    audio = recognizer.listen(source)
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{index}: {name}")



print("🧠 Mevcut mikrofonlar:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{index}: {name}")
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
mp_drawing = mp.solutions.drawing_utils## bak buraya

heart_img = cv2.imread("heart.png", cv2.IMREAD_UNCHANGED)
heart_img = cv2.resize(heart_img, (100, 100))

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

    print(f"[HEART] Index Tip Y: {index_tip.y:.2f}, Thumb Tip Y: {thumb_tip.y:.2f}, Distance: {distance:.3f}")

    return distance < 0.32  # bu aralığa göre ideal eşik!


def is_ok_pose(hand_landmarks):
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]

    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

    # Yazdırmaya devam edelim (debug için)
    #print(f"Index Tip Y: {index_tip.y:.2f}, Thumb Tip Y: {thumb_tip.y:.2f}, Distance: {distance:.3f}")

    # Eşiği 0.38 yaptık, çünkü senin gesture'ların genelde 0.33–0.36 aralığında
    return distance < 0.38 and index_tip.y > thumb_tip.y

def listen_for_keyword(expected_keyword):
    recognizer = sr.Recognizer()

    # Mikrofon indexini güvenli şekilde otomatik al
    mic_list = sr.Microphone.list_microphone_names()
    if not mic_list:
        print("[ERROR] Hiçbir mikrofon bulunamadı.")
        return False

    print("🧠 Mikrofonlar:", mic_list)
    device_index = 0  # Gerekirse elle doğru index'i buraya yaz

    try:
        with sr.Microphone(device_index=device_index) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[INFO] Lütfen konuşun...")
            audio = recognizer.listen(source, phrase_time_limit=5)
            text = recognizer.recognize_google(audio, language="tr-TR")
            print(f"[INFO] Algılanan ses: {text}")
            return expected_keyword.lower() in text.lower()
    except AssertionError as ae:
        print("[ERROR] Mikrofon başlatılamadı. Muhtemelen device_index hatalı.")
    except sr.UnknownValueError:
        print("[INFO] Ses anlaşılamadı.")
    except sr.RequestError as e:
        print(f"[ERROR] Google API hatası: {e}")
    except Exception as e:
        print(f"[ERROR] Genel hata: {e}")
    return False

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
    video_capture = cv2.VideoCapture(1)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print('[INFO] Webcam well started')
    print('[INFO] Detecting...')


    frame_count = 0
    expected_keyword = "merhaba"
    
    while True:
        
        
        ret, frame = video_capture.read()
        if not ret or frame is None:
            print("[ERROR] Kameradan görüntü alınamadı.")
            break
        if frame_count % 5 == 0:
           easy_face_reco(frame, known_face_encodings, known_face_names)
        
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            hand_poses = [hand for hand in result.multi_hand_landmarks]

            # Her eli çiz (iskelet çizimi) 
            #for hand_landmarks in hand_poses:
                #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            #bence bu çizim kısmı güzel durmuyor ondan o çizimleri sildim 

            # Kalp gesture için: iki el varsa ve ikisi de kalp pozunda ise
            if len(hand_poses) == 2:
                poses = [is_heart_pose(hand) for hand in hand_poses]
                if all(poses):
                    h, w, _ = frame.shape
                    x, y = w // 2 - 40, h // 2 - 40
                    for c in range(3):  # RGB
                        frame[y:y+100, x:x+100, c] = (
                            heart_img[:, :, c] * (heart_img[:, :, 3] / 255.0) +
                            frame[y:y+100, x:x+100, c] * (1.0 - heart_img[:, :, 3] / 255.0)
                        )
                    cv2.putText(frame, "Kalp!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # OK gesture için: tek el varsa ve poz doğruysa
            elif len(hand_poses) == 1:
                if is_ok_pose(hand_poses[0]):
                    h, w, _ = frame.shape
                    x, y = w // 2 - 40, h // 2 - 40
                    for c in range(3):  # RGB
                        frame[y:y+80, x:x+80, c] = (
                            ok_img[:, :, c] * (ok_img[:, :, 3] / 255.0) +
                            frame[y:y+80, x:x+80, c] * (1.0 - ok_img[:, :, 3] / 255.0)
                        )
                    cv2.putText(frame, "Tamam!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            # Sesle anahtar kelime doğrulama
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if not keyword_checking:
             threading.Thread(target=threaded_keyword_check, args=(expected_keyword,), daemon=True).start()
        if keyword_success:
           cv2.putText(frame, "Correct password.Successfully logged in", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        elif keyword_checking:
          cv2.putText(frame, "Listening for voice...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        # Kamera görüntüsünü göster
        cv2.imshow('Easy Facial Recognition App', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Stopping System')
    video_capture.release()
    cv2.destroyAllWindows()
    hands.close()
