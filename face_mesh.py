import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def get_landmarks(image_path, draw=False):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []

        h, w, _ = image.shape

        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

            # 🔥 GAMBAR TITIK
            if draw:
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        # 🔥 TAMPILKAN GAMBAR
        if draw:
            cv2.imshow("Landmarks", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return landmarks