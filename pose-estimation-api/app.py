from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import time
import os
import uuid
from werkzeug.utils import secure_filename
from squat_detector import SquatDetector
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# 설정
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detectors = {}
processing_sessions = {}

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
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        session_id = str(uuid.uuid4())
        processing_sessions[session_id] = {
            "status": "processing",
            "file_path": file_path,
            "start_time": time.time()
        }

        # 백그라운드에서 비디오 처리 시작
        threading.Thread(
            target=process_video_streaming,
            args=(file_path, session_id),
            daemon=True
        ).start()

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "data": "Realtime video processing started"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/session/<session_id>/status', methods=['GET'])
def get_session_status(session_id):
    if session_id in processing_sessions:
        return jsonify({
            "status": "success",
            "data": processing_sessions[session_id]
        })
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/session/<session_id>/stop', methods=['POST'])
def stop_session(session_id):
    if session_id in processing_sessions:
        processing_sessions[session_id]["status"] = "stopped"
        return jsonify({"status": "success", "message": "Session stopped"})
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_id in processing_sessions:
        # 파일 정리
        file_path = processing_sessions[session_id].get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        del processing_sessions[session_id]
        if session_id in detectors:
            del detectors[session_id]

        return jsonify({"status": "success", "message": "Session deleted"})
    return jsonify({"error": "Session not found"}), 404

# WebSocket
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to squat analysis server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('join_session')
def handle_join_session(data):
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        emit('joined_session', {'session_id': session_id})
        print(f'Client joined session: {session_id}')

@socketio.on('leave_session')
def handle_leave_session(data):
    session_id = data.get('session_id')
    if session_id:
        leave_room(session_id)
        emit('left_session', {'session_id': session_id})
        print(f'Client left session: {session_id}')

def process_video_streaming(video_path, session_id):
    try:
        detector = SquatDetector()
        detectors[session_id] = detector

        def websocket_callback(event_type, data):
            """SquatDetector에서 호출할 콜백 함수"""
            # 세션 ID 추가
            data["session_id"] = session_id

            # WebSocket으로 전송
            socketio.emit(event_type, data, room=session_id)

        def stop_check():
            """중지 상태 확인 함수"""
            return processing_sessions.get(session_id, {}).get("status") == "stopped"

        # 처리 시작 시간 기록
        start_time = time.time()

        # SquatDetector의 스트리밍 메소드 호출
        result = detector.process_video_streaming(
            video_path,
            callback=websocket_callback,
            stop_callback=stop_check
        )

        # 최종 완료 알림에 처리 시간 추가
        socketio.emit('analysis_complete', {
            "session_id": session_id,
            "final_squat_count": result["final_squat_count"],
            "final_state": result["final_state"],
            "total_frames": result["total_frames"],
            "processing_time": time.time() - start_time
        }, room=session_id)

        # 세션 상태 업데이트
        if session_id in processing_sessions:
            processing_sessions[session_id]["status"] = "completed"

    except Exception as e:
        # 에러 발생 시 클라이언트에 알림
        socketio.emit('analysis_error', {
            "session_id": session_id,
            "error": str(e)
        }, room=session_id)
        print(f"Error in session {session_id}: {str(e)}")

    finally:
        # 파일 정리
        if os.path.exists(video_path):
            os.remove(video_path)

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)