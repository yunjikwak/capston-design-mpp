from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from squat_detector import SquatDetector

app = Flask(__name__)
CORS(app)

# 설정
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detectors = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Squat Detection API is running"
    })

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # 파일 저장
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # 스쿼트 분석
        detector = SquatDetector()
        result = detector.process_video(file_path)

        # 임시 파일 삭제
        os.remove(file_path)

        return jsonify({
            "status": "success",
            "data": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze/realtime', methods=['POST'])
def analyze_realtime():
    try:
        data = request.get_json()
        session_id = data.get('session_id', str(uuid.uuid4()))
        landmarks_data = data.get('landmarks')

        if not landmarks_data:
            return jsonify({"error": "No landmarks data provided"}), 400

        if session_id not in detectors:
            detectors[session_id] = SquatDetector()
        detector = detectors[session_id]

        class MockLandmark:
            def __init__(self, x, y, z, visibility):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
        mock_landmarks = [MockLandmark(**lm) for lm in landmarks_data]

        analysis_result = detector.process_frame(mock_landmarks)

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "data": analysis_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/session/reset/<session_id>', methods=['POST'])
def reset_session(session_id):
    if session_id in detectors:
        detectors[session_id].reset_state()
        return jsonify({
            "status": "success",
            "message": "Session reset"})
    return jsonify({"error": "Session not found"}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)