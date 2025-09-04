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

def open_camera(device_index: int, backend: str = None):
    if backend == "dshow":
        return cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if backend == "msmf":
        return cv2.VideoCapture(device_index, cv2.CAP_MSMF)
    return cv2.VideoCapture(device_index)

async def main(device_index: int, backend: str):
    session_id = "dev1"
    uri = f"ws://localhost:8000/ws/{session_id}"
    print("Connecting to", uri, "camera index:", device_index, "backend:", backend)
    async with websockets.connect(uri, max_size=8_000_000) as ws:
        cap = open_camera(device_index, backend)
        if not cap.isOpened():
            print("Camera open failed for index", device_index)
            return
        frame_id = 0
        last_recv = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed")
                break
            h_target = 480
            ratio = h_target / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1]*ratio), h_target))
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
    args = p.parse_args()
    asyncio.run(main(args.device, args.backend or None))