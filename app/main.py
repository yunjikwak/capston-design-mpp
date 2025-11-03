from typing import Dict, Optional, List
import base64
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from services.squat_logic import SquatSession, run_pose_inference_bgr
import logging

# 테스트용 로그 찍기
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("squat")

app = FastAPI(title="Squat AI Service", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 저장
sessions: Dict[str, SquatSession] = {}

# 좌/우 주요 인덱스(visibility 합으로 side 자동 선택)
LEFT_IDX  = {"HIP": 23, "KNEE": 25, "SHOULDER": 11}
RIGHT_IDX = {"HIP": 24, "KNEE": 26, "SHOULDER": 12}
SIDE_VIS_MARGIN = 0.10  # 좌/우 visibility 합 차이가 이 값 이상일 때만 결정

def _sum_vis(lm, idx): # visibility 합 계산
    vals = []
    for k in ("HIP", "KNEE", "SHOULDER"):
        i = idx[k]
        if 0 <= i < len(lm):
            # visibility 값이 없으면 0.0으로 설정
            v = getattr(lm[i], "visibility", 0.0) or 0.0
            vals.append(v)
    return sum(vals)

def pick_side_auto(lm):
    l_sum = _sum_vis(lm, LEFT_IDX)
    r_sum = _sum_vis(lm, RIGHT_IDX)
    if abs(l_sum - r_sum) < SIDE_VIS_MARGIN:
        # 큰 차이가 없으면 이전값 유지하도록 호출측에서 처리
        return "left" if l_sum >= r_sum else "right"
    return "left" if l_sum > r_sum else "right"

def pack_response(
    s: SquatSession,
    out: dict,
    frame_id: Optional[object] = None,
    lm: Optional[List[object]] = None,
    include_landmarks: bool = False,
    include_debug: bool = False
):
    resp = {
        "frame_id": frame_id,
        "state": out.get("state", s.state),
        "squat_count": out.get("squat_count", s.squat_count),
        "feedback": out.get("feedback", ""),
        "side": getattr(s, "side", None),
        "avg_score": out.get("avg_score", getattr(s, "avg_score", None)),
    }
    # 반복 완료 시에만 존재하는 필드들
    for k in ("score", "grade", "breakdown", "recent"):
        if k in out:
            resp[k] = out[k]

    # 랜드마크
    if include_landmarks and lm is not None:
        resp["landmarks"] = [
            {"x": p.x, "y": p.y, "z": getattr(p, "z", 0.0), "visibility": getattr(p, "visibility", 0.0)}
            for p in lm
        ]

    if include_debug and lm is not None and hasattr(s, "_joints_visible"):
        try:
            ok, vis_map = s._joints_visible(lm, getattr(s, "side", "left"))
            resp.setdefault("debug", {})["vis"] = vis_map
        except Exception:
            pass

    return resp

@app.get("/api/health")
def health(): # 상태 확인
    return {"status": "ok"}

@app.post("/api/session")
def create_session(side: str = "auto"): # 세션 생성
    s = SquatSession(side=side)
    sessions[s.id] = s
    return {"session_id": s.id, "side": side}

@app.get("/api/session/{session_id}")
def session_status(session_id: str): # 세션 상태 확인
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(404, "not found")
    return {
        "session_id": s.id,
        "state": s.state,
        "squat_count": s.squat_count,
        "avg_score": s.avg_score,
        "side": getattr(s, "side", None),
    }

@app.delete("/api/session/{session_id}")
def delete_session(session_id: str): # 세션 삭제
    sessions.pop(session_id, None)
    return {"deleted": session_id}

@app.post("/api/session/{session_id}/frame") # 프레임 업로드 / 서버 테스트용
async def upload_frame(
    session_id: str,
    file: UploadFile = File(...),
    include_landmarks: bool = Query(False),
    include_debug: bool = Query(False)
):
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
        out = {"state": s.state, "squat_count": s.squat_count, "feedback": "NoPose"}
        return pack_response(s, out)

    lm = res.pose_landmarks.landmark
    # side 자동 선택
    if getattr(s, "side", None) in (None, "auto"):
        chosen = pick_side_auto(lm)
        try:
            s.side = chosen
        except Exception:
            pass

    out = s.process_landmarks(lm)
    return pack_response(s, out, lm=lm, include_landmarks=include_landmarks, include_debug=include_debug)

@app.websocket("/ws/{session_id}")
async def ws_stream(ws: WebSocket, session_id: str, debug: bool = False):
    await ws.accept()
    # 세션 확인
    if session_id not in sessions:
        await ws.send_json({
            "status": 404,
            "code": "SESSION_NOT_FOUND",
            "message": "세션을 찾을 수 없음"
        })
        await ws.close(code=1008)
        logger.warning(f"[{session_id}] rejected: session not found")
        return

    s = sessions[session_id] # 세션 객체 가져오기
    logger.info(f"[{session_id}] websocket connected")
    try:
        while True:
            msg = await ws.receive_json()
            if "image_base64" not in msg:
                await ws.send_json({
                    "status": 400,
                    "code": "NO_IMAGE",
                    "message": "image_base64 필드 누락"
                })
                continue

            # 이미지 디코드
                # Base64 → 바이너리 → OpenCV 이미지
            raw = base64.b64decode(msg["image_base64"])
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_json({
                    "status": 400,
                    "code": "DECODE_FAILED",
                    "message": "JPEG 디코드 실패"
                })
                continue

            res = run_pose_inference_bgr(frame)
            # 랜드마크 없으면 예외 처리
            if not getattr(res, "pose_landmarks", None):
                await ws.send_json({
                    "frame_id": msg.get("frame_id"),
                    "squat_count": s.squat_count,
                    "feedback": "NoPose",
                    "avg_score": s.avg_score
                })
                continue

            # 랜드마크 가져오기
            lm = res.pose_landmarks.landmark
            # side 자동 선택
            if getattr(s, "side", None) in (None, "auto"):
                chosen = pick_side_auto(lm)
                if getattr(s, "side", None) != chosen:
                    logger.info(f"[{session_id}] side auto -> {chosen}")
                try:
                    s.side = chosen
                except Exception:
                    pass

            # 로깅용 - 상태 변화 및 카운트 증가
            prev_state, prev_cnt = s.state, s.squat_count
            out = s.process_landmarks(lm) # 로직 수행
            if s.state != prev_state:
                logger.info(f"[{session_id}] state {prev_state} -> {s.state} fb={out.get('feedback','')}")
            if s.squat_count > prev_cnt:
                logger.info(f"[{session_id}] count={s.squat_count} score={out.get('score')} grade={out.get('grade')} avg={getattr(s,'avg_score',None)}")

            # 기본 응답
            resp = {
                "squat_count": s.squat_count,
                "feedback": out.get("feedback", ""),
                "avg_score": s.avg_score
            }
            # 완료 시 추가
            if "score" in out:
                resp["score"] = out["score"]
                resp["grade"] = out["grade"]

            # 디버그 모드면 landmark 추가
            if debug and lm is not None:
                resp["landmarks"] = [
                    {"x": p.x, "y": p.y, "z": getattr(p, "z", 0.0),
                    "visibility": getattr(p, "visibility", 0.0)}
                    for p in lm
                ]

            await ws.send_json(resp)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)