from typing import Dict
import base64
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.squat_logic import SquatSession, run_pose_inference_bgr

app = FastAPI(title="Squat AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, SquatSession] = {}

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/session")
def create_session(side: str = "left"):
    s = SquatSession(side=side)
    sessions[s.id] = s
    return {"session_id": s.id}

@app.get("/api/session/{session_id}")
def session_status(session_id: str):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, "not found")
    return {
        "session_id": s.id,
        "state": s.state,
        "squat_count": s.squat_count,
        "avg_score": s.avg_score,
        "recent": s.recent_rep_feedback
    }

@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"deleted": session_id}

@app.post("/api/session/{session_id}/frame")
async def upload_frame(session_id: str, file: UploadFile = File(...)):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, "not found")
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "decode failed")
    res = run_pose_inference_bgr(frame)
    if not res.pose_landmarks:
        return {"state": s.state, "squat_count": s.squat_count, "feedback": "NoPose"}
    return s.process_landmarks(res.pose_landmarks.landmark)

@app.websocket("/ws/{session_id}")
async def ws_stream(ws: WebSocket, session_id: str):
    await ws.accept()
    if session_id not in sessions:
        sessions[session_id] = SquatSession()
    s = sessions[session_id]
    try:
        while True:
            msg = await ws.receive_json()
            if "image_base64" in msg:
                raw = base64.b64decode(msg["image_base64"])
                arr = np.frombuffer(raw, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    await ws.send_json({"error": "decode_failed"})
                    continue
                res = run_pose_inference_bgr(frame)
                if not res.pose_landmarks:
                    await ws.send_json({"state": s.state, "squat_count": s.squat_count, "feedback": "NoPose", "landmarks": []})
                    continue
                out = s.process_landmarks(res.pose_landmarks.landmark)
                # landmarks를 JSON으로 변환해서 포함
                out["landmarks"] = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in res.pose_landmarks.landmark
                ]
                out["frame_id"] = msg.get("frame_id")
                await ws.send_json(out)
            else:
                await ws.send_json({"error": "unknown_payload"})
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000)