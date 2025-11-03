import argparse, asyncio, websockets, base64, json, cv2, time

def draw_landmarks(frame, landmarks, show_index=False):
    h, w = frame.shape[:2]
    for i, lm in enumerate(landmarks):
        try:
            x = int(lm['x'] * w)
            y = int(lm['y'] * h)
        except Exception:
            continue
        cv2.circle(frame, (x, y), 6, (0,255,0), -1)
        cv2.circle(frame, (x, y), 8, (0,0,0), 1)
        if show_index:
            cv2.putText(frame, str(i), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

def open_camera(device_index: int, backend: str = None, video_file: str = None):
    if video_file:
        # 영상 파일 사용
        cap = cv2.VideoCapture(video_file)
        print(f"📹 영상 파일 로드: {video_file}")
    else:
        # 카메라 사용
        if backend == "dshow":
            cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        elif backend == "msmf":
            cap = cv2.VideoCapture(device_index, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(device_index)

        # 1080p 시도
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # 실제로 설정된 해상도 확인
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"요청 해상도: 1920x1080")
        print(f"실제 해상도: {actual_w}x{actual_h}")

    return cap

async def main(device_index: int, backend: str, video_file: str = None):
    # Azure 배포 서버 URL
    base_url = "squat-api.blackmoss-f506213d.koreacentral.azurecontainerapps.io"
    session_id = "65b184d0-698a-4127-916d-a724932ccef3" # 변경하기
    # 연결 (debug=true로 landmark 받기)
    uri = f"wss://{base_url}/ws/{session_id}?debug=true"
    print("Connecting to", uri, "camera index:", device_index, "backend:", backend)
    async with websockets.connect(uri, max_size=8_000_000) as ws:
        cap = open_camera(device_index, backend, video_file)
        if not cap.isOpened():
            print("Camera open failed for index", device_index)
            return
        frame_id = 0
        last_recv = time.time()
        last_time = time.time()
        FPS_LIMIT = 10  # 초당 10프레임
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed")
                break

            # FPS 제한
            current_time = time.time()
            if current_time - last_time < 1.0 / FPS_LIMIT:
                continue
            last_time = current_time
            max_dimension = 1080
            h, w = frame.shape[:2]

            if h > max_dimension or w > max_dimension:
                if h > w:
                    ratio = max_dimension / h
                else:
                    ratio = max_dimension / w
                frame = cv2.resize(frame, (int(w*ratio), int(h*ratio)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                print("JPEG encode failed")
                break
            b64 = base64.b64encode(buf).decode()
            await ws.send(json.dumps({"frame_id": frame_id, "image_base64": b64}))
            try:
                resp = await ws.recv()
            except Exception as e:
                print("recv error:", e)
                break
            last_recv = time.time()
            try:
                resp_json = json.loads(resp)
            except Exception:
                resp_json = {}

            # 로그 출력 (핵심 정보만)
            if "landmarks" in resp_json and resp_json["landmarks"]:
                # landmark가 있을 때는 핵심 정보만
                core_info = {k: v for k, v in resp_json.items() if k != "landmarks"}
                print(f"[Frame {frame_id}] 응답:", core_info)

                # 모든 상태별 디버그 정보
                state = core_info.get("state", "")
                extra = core_info.get("extra", {})
                breakdown = core_info.get("breakdown", {})
                vis_info = core_info.get("vis", {})

                if state == "START":
                    print(f"  └─ START 상태: 엉덩이-무릎 거리 체크 중")
                    print(f"  └─ hip_y: {extra.get('hip_y', 'N/A')}, knee_y: {extra.get('knee_y', 'N/A')}")
                    print(f"  └─ hip_knee_gap: {extra.get('hip_knee_gap', 'N/A')}, ENTER_SIT_GAP: 0.06")
                    print(f"  └─ 전환 조건: gap < 0.06 = {extra.get('hip_knee_gap', 999) < 0.06 if 'hip_knee_gap' in extra else 'N/A'}")

                elif state == "SIT":
                    # 기본 정보
                    print(f"  └─ bottom_locked: {extra.get('bottom', 'NO')}, sit_frames: {breakdown.get('sit_frames', 0)}")

                    # visibility 정보
                    if vis_info:
                        print(f"  └─ visibility: {vis_info}")

                    # 스쿼트 관련 상세 정보
                    if breakdown:
                        print(f"  └─ depth_ratio: {breakdown.get('depth_ratio', 'N/A')}, knee_angle: {breakdown.get('knee_angle', 'N/A')}")
                        print(f"  └─ knee%: {breakdown.get('knee%', 'N/A')}, back%: {breakdown.get('back%', 'N/A')}, depth%: {breakdown.get('depth%', 'N/A')}")

                    # 속도 관련 정보 (extra에서)
                    if "inst_vel" in extra:
                        print(f"  └─ inst_vel: {extra.get('inst_vel', 'N/A')}, UP_VEL_THRESH: -0.0025")

                    # 상태 전환 조건 체크
                    print(f"  └─ 조건 체크: bottom_locked={extra.get('bottom') == 'OK'}, vel_ok={extra.get('inst_vel', 0) < -0.0025 if 'inst_vel' in extra else 'N/A'}")

                elif state == "RISING":
                    print(f"  └─ RISING 상태: 일어서는 중")
                    if "progress" in extra:
                        print(f"  └─ progress: {extra.get('progress', 'N/A')}")

                elif state == "STAND":
                    print(f"  └─ STAND 상태: 서있는 상태")
                    if "score" in core_info:
                        print(f"  └─ 완료! 점수: {core_info.get('score', 'N/A')}, 등급: {core_info.get('grade', 'N/A')}")
            else:
                # landmark가 없을 때는 전체 출력
                print(f"[Frame {frame_id}] 응답:", resp_json)

            if "landmarks" in resp_json and resp_json["landmarks"]:
                draw_landmarks(frame, resp_json["landmarks"], show_index=False)
            else:
                fb = resp_json.get("feedback") or resp_json.get("error") or ""
                cv2.putText(frame, f"Server: {fb}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Camera Preview (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_id += 1
            if time.time() - last_recv > 5.0:
                print("No response from server for 5s, exiting")
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", "-d", type=int, default=0, help="camera device index (0,1,2...)")
    p.add_argument("--backend", "-b", choices=["dshow","msmf",""], default="dshow", help="Windows backend (dshow/msmf). empty for default")
    p.add_argument("--video", "-v", type=str, help="video file path instead of camera")
    args = p.parse_args()
    asyncio.run(main(args.device, args.backend or None, args.video))