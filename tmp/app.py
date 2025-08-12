from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from squat_detector import SquatDetector
import json

app = Flask(__name__)
# CORS(app)

# 설정
UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 업로드 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def allowed_file(filename):
#     """허용된 파일 확장자 확인"""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """서버 상태 확인"""
#     return jsonify({
#         "status": "healthy",
#         "message": "Squat Detection API is running"
#     })

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    try:
        # 파일 확인
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # if not allowed_file(file.filename):
        #     return jsonify({"error": "Invalid file format"}), 400

        # 파일 저장
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # 스쿼트 분석
        detector = SquatDetector()
        analysis_result = detector.process_video(file_path)

        # 임시 파일 삭제
        os.remove(file_path)

        return jsonify({
            "status": "success",
            "data": analysis_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze/realtime', methods=['POST'])
def analyze_realtime():
    """실시간 스쿼트 분석 (단일 프레임)"""
    try:
        data = request.get_json()

        if 'landmarks' not in data:
            return jsonify({"error": "No landmarks data provided"}), 400

        # 스쿼트 감지기 초기화 (세션 관리 필요)
        detector = SquatDetector()

        # 랜드마크 데이터 처리
        landmarks = data['landmarks']
        feedback = detector.process_state(landmarks)

        return jsonify({
            "status": "success",
            "data": {
                "state": detector.state,
                "squat_count": detector.squat_count,
                "feedback": feedback
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_detector():
    """스쿼트 감지기 상태 초기화"""
    try:
        # 실제 구현에서는 세션 관리가 필요
        return jsonify({
            "status": "success",
            "message": "Detector state reset"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """파일 크기 초과 에러 처리"""
    return jsonify({"error": "File too large"}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
