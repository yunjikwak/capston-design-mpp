import cv2
import mediapipe as mp
import numpy as np
import time

class SquatDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 설정 값
        self.TIME_LIMIT = 10
        self.BACK_STRAIGHTNESS_THRESHOLD = 0.15

        # 상태 변수
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

        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)

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

        return feedback

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        results = []

        with self.mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # BGR → RGB 변환
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                pose_results = pose.process(image)
                feedback = ""

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    feedback = self.process_state(landmarks)

                # 결과 저장
                frame_result = {
                    "frame_number": len(results),
                    "state": self.state,
                    "squat_count": self.squat_count,
                    "feedback": feedback,
                    "landmarks_detected": pose_results.pose_landmarks is not None
                }
                results.append(frame_result)

        cap.release()
        return {
            "total_frames": len(results),
            "final_squat_count": self.squat_count,
            "final_state": self.state,
            "frames": results
        }
