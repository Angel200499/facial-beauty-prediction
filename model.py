import numpy as np
import pickle

# ===============================
# 🔹 SYMMETRY
# ===============================
def calculate_symmetry(landmarks):
    landmarks = np.array(landmarks)

    mid_x = np.mean(landmarks[:, 0])
    symmetry_diff = []

    for (x, y) in landmarks:
        mirrored_x = 2 * mid_x - x

        distances = np.sqrt(
            (landmarks[:, 0] - mirrored_x) ** 2 +
            (landmarks[:, 1] - y) ** 2
        )

        closest = np.min(distances)
        symmetry_diff.append(closest)

    symmetry_score = np.mean(symmetry_diff)

    score = 10 - (symmetry_score * 50)

    return max(1, min(10, score))


# ===============================
# 🔹 PROPORTION
# ===============================
def calculate_proportion(landmarks):
    landmarks = np.array(landmarks)

    try:
        # titik penting MediaPipe
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_face = landmarks[234]
        right_face = landmarks[454]

        eye_distance = np.linalg.norm(left_eye - right_eye)
        face_width = np.linalg.norm(left_face - right_face)

        ratio = eye_distance / face_width

        score = 10 - abs(ratio - 0.3) * 20

        return max(1, min(10, score))

    except:
        return 5  # fallback kalau error


# ===============================
# 🔹 RULE-BASED COMBINATION
# ===============================
def final_score(symmetry, proportion):
    return (0.6 * symmetry) + (0.4 * proportion)


# ===============================
# 🔹 MACHINE LEARNING
# ===============================
try:
    with open("ml_model.pkl", "rb") as f:
        ml_model = pickle.load(f)
except:
    ml_model = None


def predict_ml(symmetry, proportion):
    if ml_model is None:
        return final_score(symmetry, proportion)

    try:
        pred = ml_model.predict([[symmetry, proportion]])[0]
        return max(1, min(10, pred))
    except:
        return final_score(symmetry, proportion)