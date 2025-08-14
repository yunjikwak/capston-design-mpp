import cv2
import mediapipe as mp
import numpy as np
import time

class SquatDetector:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.TIME_LIMIT = 10
        self.BACK_STRAIGHTNESS_THRESHOLD = 0.15
        self.reset_state()

    def reset_state(self):
        self.squat_count = 0
        self.initial_hip_y = None
        self.state = "START"
        self.sit_start_time = None

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ab = a - b
        bc = c - b

        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def check_knee_angle(self, landmarks):
        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # 무릎 각도 계산
        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)

        # 피드백 생성
        if knee_angle < 90:
            feedback = f"Good! Knee angle: {knee_angle:.2f} degrees"
            is_bent_correctly = True
        else:
            feedback = f"Bad! Knee angle: {knee_angle:.2f} degrees"
            is_bent_correctly = False

        return feedback, is_bent_correctly

    def process_state(self, landmarks):
        feedback = ""
        left_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
        left_knee_y = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y
        left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

        # 처음 자세의 엉덩이 위치 저장
        if self.initial_hip_y is None:
            self.initial_hip_y = left_hip_y

        # 허리 펴짐 정도 계산
        back_straightness = abs(left_hip_y - left_shoulder_y)

        # 상태 전환 로직
        if self.state == "START":
            if back_straightness < self.BACK_STRAIGHTNESS_THRESHOLD:
                feedback = "Good posture while standing!"
            else:
                feedback = "Bad posture! Straighten your back."
            if abs(left_hip_y - left_knee_y) < 0.05:
                self.state = "SIT"
                self.sit_start_time = time.time()
        elif self.state == "SIT":
            feedback, is_bent_correctly = self.check_knee_angle(landmarks)
            if time.time() - self.sit_start_time > self.TIME_LIMIT:
                self.state = "START"
                feedback = "Time exceeded! Resetting to START."
            elif left_hip_y < left_knee_y - 0.05 and (left_hip_y <= self.initial_hip_y + 0.05):
                self.state = "STAND"
                self.squat_count += 1
        elif self.state == "STAND":
            if back_straightness < self.BACK_STRAIGHTNESS_THRESHOLD:
                feedback = "Good posture while standing!"
            else:
                feedback = "Bad posture! Straighten your back."
            if left_hip_y >= left_knee_y - 0.05:
                self.state = "SIT"
                self.sit_start_time = time.time()

        return {
            "state": self.state,
            "squat_count": self.squat_count,
            "feedback": feedback
        }

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        results = []

        # FPS 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_skip = int(fps / 5)  # 5fps로 샘플링 (6배 빨라짐)

        with self.mp_pose.Pose(
            model_complexity=1, # 0: lite, 1: full, 2: heavy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            frame_number = 0
            processed_frames = 0

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                 # 프레임 스킵
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue

                # BGR → RGB 변환
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(image)
                # feedback = ""
                frame_data = {
                    "frame_number": frame_number,
                    "landmarks_detected": False,
                    "landmarks": None,
                    "analysis": None
                }
                essential_landmarks = [
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_ANKLE,
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER
                ]

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    # feedback = self.process_state(landmarks)
                    keypoints = []
                    # for landmark in landmarks:
                    #     keypoints.append({
                    #         "x": landmark.x,
                    #         "y": landmark.y,
                    #         "z": landmark.z,
                    #         "visibility": landmark.visibility
                    #     })
                    for idx in essential_landmarks:
                        landmark = landmarks[idx.value]
                        keypoints.append({
                            "name": idx.name,
                            "x": landmark.x,
                            "y": landmark.y
                        })

                    frame_data.update({
                        "landmarks_detected": True,
                        "landmarks": keypoints,
                        "analysis": self.process_state(landmarks)
                    })

                results.append(frame_data)
                frame_number += 1
                processed_frames += 1

                # 진행률 출력 (디버깅용)
                if processed_frames % 10 == 0:
                    print(f"Processed {processed_frames} frames...")

        cap.release()
        return {
            "total_frames": len(results),
            "processed_frames": processed_frames,
            "final_squat_count": self.squat_count,
            "final_state": self.state,
            "frames": results
        }

    def process_frame(self, landmarks):
        return self.process_state(landmarks)

    def process_video_streaming(self, video_path, callback=None, stop_callback=None):
        cap = cv2.VideoCapture(video_path)

        # FPS 정보
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / fps
        frame_skip = int(fps / 5)  # 5fps로 샘플링

        # 시작 정보
        if callback:
            callback('analysis_started', {
                "fps": fps,
                "message": "Video analysis started"
            })

        with self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            frame_number = 0
            processed_frames = 0

            while cap.isOpened():
                # 중지 확인
                if stop_callback and stop_callback():
                    break

                success, image = cap.read()
                if not success:
                    break

                # 프레임 스킵
                if frame_number % frame_skip != 0:
                    frame_number += 1
                    continue

                # 실제 비디오 재생 속도에 맞춰 지연
                import time
                time.sleep(frame_delay * frame_skip)

                # BGR → RGB 변환
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(image)

                frame_data = {
                    "frame_number": frame_number,
                    "timestamp": frame_number / fps,
                    "landmarks_detected": False,
                    "analysis": None
                }

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark

                    # 핵심 랜드마크만 추출
                    essential_landmarks = [
                        self.mp_pose.PoseLandmark.LEFT_KNEE,
                        self.mp_pose.PoseLandmark.LEFT_HIP,
                        self.mp_pose.PoseLandmark.LEFT_ANKLE,
                        self.mp_pose.PoseLandmark.LEFT_SHOULDER
                    ]

                    keypoints = []
                    for idx in essential_landmarks:
                        landmark = landmarks[idx.value]
                        keypoints.append({
                            "name": idx.name,
                            "x": landmark.x,
                            "y": landmark.y
                        })

                    frame_data.update({
                        "landmarks_detected": True,
                        "landmarks": keypoints,
                        "analysis": self.process_state(landmarks)
                    })

                # 콜백으로 프레임 데이터 전송
                if callback:
                    callback('frame_analysis', {"data": frame_data})

                frame_number += 1
                processed_frames += 1

                # 진행률 업데이트 (10프레임마다)
                if processed_frames % 10 == 0 and callback:
                    callback('progress_update', {
                        "processed_frames": processed_frames,
                        "total_estimated": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_skip)
                    })

        cap.release()

        # 완료 알림
        if callback:
            callback('analysis_complete', {
                "final_squat_count": self.squat_count,
                "final_state": self.state,
                "total_frames": processed_frames
            })

        return {
            "total_frames": processed_frames,
            "final_squat_count": self.squat_count,
            "final_state": self.state
        }