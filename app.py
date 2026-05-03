from flask import Flask, request, jsonify
from face_mesh import get_landmarks
from model import calculate_symmetry, calculate_proportion, final_score, predict_ml
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({"error": "Tidak ada file dikirim"})

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "File kosong"})

    path = "temp.jpg"
    file.save(path)

    landmarks = get_landmarks(path)

    if landmarks is None:
        return jsonify({"error": "Wajah tidak terdeteksi"})

    # Feature Extraction
    sym = calculate_symmetry(landmarks)
    prop = calculate_proportion(landmarks)

    # Rule-based
    rule_score = final_score(sym, prop)

    # ML
    ml_score = predict_ml(sym, prop)

    kategori = "Menarik" if ml_score >= 7 else "Cukup"

    return jsonify({
        "beauty_score_ml": round(ml_score, 2),
        "beauty_score_rule": round(rule_score, 2),
        "symmetry": round(sym, 2),
        "proportion": round(prop, 2),
        "kategori": kategori,
        "info": "Analisis berbasis simetri, proporsi, dan Machine Learning"
    })

if __name__ == '__main__':
    app.run(debug=True)