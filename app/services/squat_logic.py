import time, uuid, math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import mediapipe as mp
import cv2

# ---------------- Hyper Parameters ----------------
MIN_VIS = 0.7
ENTER_SIT_GAP = 0.06
DEPTH_MIN = 0.07
BOTTOM_MIN_FRAMES = 4
VEL_WINDOW = 4
UP_VEL_THRESH = -0.0025
KNEE_GAP = 0.07
STAND_HIP_MARGIN = 0.03
KNEE_MIN_OK = 70
KNEE_MAX_OK = 110
IDEAL_DEPTH_RATIO = 1.0
TARGET_SIT_FRAMES = 6
W_KNEE = 0.35
W_BACK = 0.25
W_DEPTH = 0.25
W_CTRL = 0.15
BACK_USE_TRUNK_ANGLE = True
TRUNK_GOOD_EXTRA = 5.0
TRUNK_BAD_EXTRA  = 20.0
TRUNK_EMA_ALPHA  = 0.25
TIME_LIMIT = 10.0
BACK_STRAIGHTNESS_THRESHOLD = 0.15
BACK_DEV_TOL = 0.40
KNEE_ANKLE_ALIGNMENT_THRESHOLD = 0.05

mp_pose = mp.solutions.pose
POSE_INSTANCE = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SIDE_IDX = {
    "left": {
        "HIP": mp_pose.PoseLandmark.LEFT_HIP.value,
        "KNEE": mp_pose.PoseLandmark.LEFT_KNEE.value,
        "ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE.value,
        "SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER.value
    },
    "right": {
        "HIP": mp_pose.PoseLandmark.RIGHT_HIP.value,
        "KNEE": mp_pose.PoseLandmark.RIGHT_KNEE.value,
        "ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        "SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    }
}

def clamp(v, lo, hi): return max(lo, min(hi, v))

def classify_grade(score: float) -> str:
    if score >= 85: return "Perfect"
    if score >= 70: return "Good"
    if score >= 55: return "Fair"
    return "Poor"

def compute_trunk_angle(lm, side):
    ids = SIDE_IDX[side]
    hip = lm[ids["HIP"]]; shoulder = lm[ids["SHOULDER"]]
    dx = shoulder.x - hip.x
    dy = shoulder.y - hip.y
    n = (dx*dx + dy*dy) ** 0.5
    if n < 1e-6: return 0.0
    cosv = max(-1.0, min(1.0, dy / n))
    return math.degrees(math.acos(cosv))

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ab = a - b; bc = c - b
    cosine = (ab @ bc) / (np.linalg.norm(ab)*np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cosine, -1, 1)))

def check_knee_angle(landmarks, side):
    ids = SIDE_IDX[side]
    hip = [landmarks[ids["HIP"]].x, landmarks[ids["HIP"]].y]
    knee = [landmarks[ids["KNEE"]].x, landmarks[ids["KNEE"]].y]
    ankle = [landmarks[ids["ANKLE"]].x, landmarks[ids["ANKLE"]].y]
    knee_angle = calculate_angle(hip, knee, ankle)
    return knee_angle

def check_knee_ankle_alignment(landmarks, side):
    ids = SIDE_IDX[side]
    knee_x = landmarks[ids["KNEE"]].x
    ankle_x = landmarks[ids["ANKLE"]].x
    # 무릎이 발목보다 앞으로 나가면 양수
    knee_forward = knee_x - ankle_x
    return abs(knee_forward) <= KNEE_ANKLE_ALIGNMENT_THRESHOLD

def run_pose_inference_bgr(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = POSE_INSTANCE.process(rgb)
    return res

def compute_rep_score(track: dict):
    if any(track.get(k) is None for k in ["knee_angle_min","hip_init","hip_bottom","knee_bottom"]):
        return 0.0, {}, "Invalid"
    knee_angle = track["knee_angle_min"]
    hip_init = track["hip_init"]
    depth_raw = track["hip_bottom"] - hip_init
    knee_span = track["knee_bottom"] - hip_init if track["knee_bottom"] > hip_init else 0
    depth_ratio = depth_raw / knee_span if knee_span > 1e-6 else 0.0

    knee_norm = clamp((KNEE_MAX_OK - knee_angle)/(KNEE_MAX_OK - KNEE_MIN_OK),0,1)

    if BACK_USE_TRUNK_ANGLE:
        base = track.get("trunk_angle_baseline")
        peak = track.get("trunk_angle_peak", base if base is not None else 0.0)
        if base is None:
            back_norm = 1.0; back_dev_out = 0.0
        else:
            extra = max(0.0, peak - base)
            if extra <= TRUNK_GOOD_EXTRA: back_norm = 1.0
            elif extra >= TRUNK_BAD_EXTRA: back_norm = 0.0
            else: back_norm = 1 - (extra - TRUNK_GOOD_EXTRA)/(TRUNK_BAD_EXTRA - TRUNK_GOOD_EXTRA)
            back_dev_out = round(extra,2)
    else:
        back_dev = track.get("back_dev_max",0.0)
        back_norm = clamp(1 - back_dev / 0.40,0,1)
        back_dev_out = round(back_dev,3)

    raw = abs(depth_ratio - IDEAL_DEPTH_RATIO)
    depth_norm = 1.0 if raw < 0.05 else clamp(1 - raw/IDEAL_DEPTH_RATIO,0,1)
    ctrl_norm = clamp(track.get("sit_frames",0)/TARGET_SIT_FRAMES,0,1)

    score = 100*(W_KNEE*knee_norm + W_BACK*back_norm + W_DEPTH*depth_norm + W_CTRL*ctrl_norm)
    grade = classify_grade(score)
    breakdown = {
        "knee%": round(knee_norm*100,1),
        "back%": round(back_norm*100,1),
        "depth%": round(depth_norm*100,1),
        "ctrl%": round(ctrl_norm*100,1),
        "knee_angle": round(knee_angle,1),
        "depth_ratio": round(depth_ratio,2),
        "sit_frames": track.get("sit_frames",0),
        "back_dev": back_dev_out
    }
    return round(score,1), breakdown, grade

class SquatSession:
    def __init__(self, side="left"):
        self.id = str(uuid.uuid4())
        self.side = side
        self.state = "START"
        self.squat_count = 0
        self.initial_hip_y = { "left": None, "right": None }
        self.sit_start_time = None
        self.rep_scores: List[float] = []
        self.rep_grades: List[str] = []
        self.rep_breakdowns: List[dict] = []
        self.avg_score = 0.0
        self.recent_rep_feedback = ""
        self.track = {
            "hip_init": None,
            "hip_bottom": None,
            "knee_bottom": None,
            "sit_frames": 0,
            "bottom_locked": False,
            "rise_frames": 0,
            "hip_history": [],
            "knee_angle_min": None,
            "back_dev_max": 0.0,
            "trunk_angle_baseline": None,
            "trunk_angle_ema": None,
            "trunk_angle_peak": None,
            "slow_frames": 0
        }

    def _joints_visible(self, lm, side, min_vis=MIN_VIS):
        ids = SIDE_IDX[side]
        needed = ["HIP","KNEE","SHOULDER"]
        vis_map = {n: lm[ids[n]].visibility for n in needed}
        return all(vis_map[n] >= min_vis for n in needed), vis_map

    def process_landmarks(self, lm):
        side = self.side
        ids = SIDE_IDX[side]
        hip_y = lm[ids["HIP"]].y
        knee_y = lm[ids["KNEE"]].y
        shoulder_y = lm[ids["SHOULDER"]].y

        # 초기 엉덩이 위치 설정
        if self.track["hip_init"] is None:
            self.track["hip_init"] = hip_y
        hip_init = self.track["hip_init"]

        # 관절 가시성 체크
        visible_ok, vis_map = self._joints_visible(lm, side)
        if not visible_ok:
            return {
                "state": self.state,
                "squat_count": self.squat_count,
                "checks": {"back": None, "knee": None, "ankle": None},
                "avg_score": self.avg_score,
                "recent": self.recent_rep_feedback,
                "message": "가시성 낮음",
                "vis": vis_map
            }

        # 엉덩이 이동 속도 계산
        hh = self.track["hip_history"]
        hh.append(hip_y)
        if len(hh) > VEL_WINDOW: hh.pop(0)
        # 순간 속도 계산
        inst_vel = hip_y - hh[-2] if len(hh) >= 2 else 0.0

        message = None
        checks = {"back": None, "knee": None, "ankle": None}

        if self.state == "START":
            # 허리 각도 측정 및 피드백
            if BACK_USE_TRUNK_ANGLE:
                # baseline 설정
                if self.track["trunk_angle_baseline"] is None:
                    ang0 = compute_trunk_angle(lm, side)
                    self.track["trunk_angle_baseline"] = ang0
                    self.track["trunk_angle_ema"] = ang0
                    self.track["trunk_angle_peak"] = ang0

                # 서 있는 자세 평가 (75~105도)
                trunk_angle = compute_trunk_angle(lm, side)
                checks["back"] = "Good" if 75 < trunk_angle < 105 else "Bad"

            # 무릎-발목 정렬 체크
            knee_ankle_aligned = check_knee_ankle_alignment(lm, side)
            checks["ankle"] = "Good" if knee_ankle_aligned else "Bad"
            # SIT 진입 조건 체크 (엉덩이와 무릎이 가까워지면)
            if abs(hip_y - knee_y) < ENTER_SIT_GAP:
                self.state = "SIT"
                self.sit_start_time = time.time()
                self.track.update({
                    # 현재 위치 -> hip_bottom, knee_bottom
                    # 측정값 초기화
                    "hip_bottom": hip_y,
                    "knee_bottom": knee_y,
                    "sit_frames": 0,
                    "bottom_locked": False, # 최소 깊이 + 최소 유지 시간 만족 여부
                    "rise_frames": 0,
                    "hip_history": [hip_y],
                    "knee_angle_min": None,
                    "back_dev_max": 0.0,
                    "slow_frames": 0
                })

        elif self.state == "SIT":
            # 무릎 각도
            knee_angle = check_knee_angle(lm, side)
            checks["knee"] = "Good" if knee_angle < 90 else "Bad"

            # 최소 무릎 각도 갱신
            if self.track["knee_angle_min"] is None or knee_angle < self.track["knee_angle_min"]:
                self.track["knee_angle_min"] = knee_angle

            # 허리 각도 측정 및 피드백
            if BACK_USE_TRUNK_ANGLE:
                ang = compute_trunk_angle(lm, side)
                ema = self.track["trunk_angle_ema"]
                self.track["trunk_angle_ema"] = ang if ema is None else (1-TRUNK_EMA_ALPHA)*ema + TRUNK_EMA_ALPHA*ang
                if self.track["trunk_angle_peak"] is None or ang > self.track["trunk_angle_peak"]:
                    self.track["trunk_angle_peak"] = ang

                # 허리 피드백
                base = self.track.get("trunk_angle_baseline")
                if base is not None:
                    extra_angle = max(0.0, ang - base)
                    threshold = (TRUNK_GOOD_EXTRA + TRUNK_BAD_EXTRA) / 2.0
                    checks["back"] = "Good" if extra_angle <= threshold else "Bad"

            # 무릎-발목 정렬 체크
            knee_ankle_aligned = check_knee_ankle_alignment(lm, side)
            checks["ankle"] = "Good" if knee_ankle_aligned else "Bad"

            # 엉덩이 위치 갱신
            if hip_y > self.track["hip_bottom"]:
                self.track["hip_bottom"] = hip_y
                self.track["knee_bottom"] = knee_y
            self.track["sit_frames"] += 1

            # 깊이 체크
            depth = self.track["hip_bottom"] - hip_init
            if (not self.track["bottom_locked"] and
                self.track["sit_frames"] >= BOTTOM_MIN_FRAMES and
                depth >= DEPTH_MIN):
                self.track["bottom_locked"] = True

            # RISING 상태 진입 조건 체크
            if self.track["bottom_locked"] and inst_vel < UP_VEL_THRESH:
                self.state = "RISING"
                self.track["rise_frames"] = 0
                self.track["hip_history"] = [hip_y]
                self.track["rising_start_hip"] = self.track["hip_bottom"]
                self.track["slow_frames"] = 0
            elif self.sit_start_time is not None and time.time() - self.sit_start_time > TIME_LIMIT:
                self.state = "START"
                message = "시간 초과. 다시 시작."

        elif self.state == "RISING":
            knee_angle = check_knee_angle(lm, side)
            if self.track["knee_angle_min"] is None or knee_angle < self.track["knee_angle_min"]:
                self.track["knee_angle_min"] = knee_angle

            back_dev = abs(hip_y - shoulder_y)
            if back_dev > self.track["back_dev_max"]:
                self.track["back_dev_max"] = back_dev

            # 허리 각도 측정 및 피드백
            if BACK_USE_TRUNK_ANGLE:
                ang = compute_trunk_angle(lm, side)
                if self.track.get("trunk_angle_ema") is None:
                    self.track["trunk_angle_ema"] = ang
                else:
                    self.track["trunk_angle_ema"] = (1-TRUNK_EMA_ALPHA)*self.track["trunk_angle_ema"] + TRUNK_EMA_ALPHA*ang
                if self.track.get("trunk_angle_peak") is None or ang > self.track["trunk_angle_peak"]:
                    self.track["trunk_angle_peak"] = ang

                # 허리 피드백 추가
                base = self.track.get("trunk_angle_baseline")
                if base is not None:
                    extra_angle = max(0.0, ang - base)
                    threshold = (TRUNK_GOOD_EXTRA + TRUNK_BAD_EXTRA) / 2.0
                    checks["back"] = "Good" if extra_angle <= threshold else "Bad"

            start_hip = self.track.get("rising_start_hip", self.track.get("hip_bottom", hip_y))
            target_y = min(hip_init + STAND_HIP_MARGIN, knee_y - KNEE_GAP)
            total_needed = start_hip - target_y
            if total_needed < 1e-6: total_needed = 1e-6
            rising_disp = start_hip - hip_y
            progress = clamp(rising_disp / total_needed, 0.0, 1.2)

            stand_ready = (hip_y < knee_y - KNEE_GAP and hip_y <= hip_init + STAND_HIP_MARGIN)
            if stand_ready:
                self.state = "STAND"
                self.squat_count += 1
                score, bd, grade = compute_rep_score(self.track)
                self.rep_scores.append(score); self.rep_grades.append(grade); self.rep_breakdowns.append(bd)
                self.avg_score = round(sum(self.rep_scores)/len(self.rep_scores),1)
                self.recent_rep_feedback = f"{score} ({grade})"
                # reset for next
                self.track.update({
                    "hip_history": [hip_y],
                    "rise_frames": 0,
                    "sit_frames": 0,
                    "bottom_locked": False,
                    "knee_angle_min": None,
                    "hip_bottom": None,
                    "knee_bottom": None,
                    "slow_frames": 0,
                    "trunk_angle_peak": self.track.get("trunk_angle_ema")
                })
                return {
                    "state": self.state,
                    "squat_count": self.squat_count,
                    "checks": {"back": None, "knee": None, "ankle": None},
                    "score": score,
                    "grade": grade,
                    "breakdown": bd,
                    "avg_score": self.avg_score,
                    "recent": self.recent_rep_feedback,
                    "message": "",
                    "progress": 1.0
                }

            if inst_vel < UP_VEL_THRESH:
                self.track["rise_frames"] += 1
            else:
                # 감속 허용: progress < 0.3 인데도 느리면 카운트다운
                if progress < 0.30:
                    self.track.setdefault("slow_frames", 0)
                    self.track["slow_frames"] += 1
                    if self.track["slow_frames"] >= 12:  # 12프레임 지속 시 실패
                        message = "자세 인식 실패"
                        self.state = "SIT"
                        self.sit_start_time = time.time()
                        self.track.update({
                            "hip_bottom": hip_y,
                            "knee_bottom": knee_y,
                            "sit_frames": 0,
                            "bottom_locked": False,
                            "rise_frames": 0,
                            "hip_history": [hip_y],
                            "knee_angle_min": None,
                            "back_dev_max": 0.0,
                            "slow_frames": 0
                        })
                        return {
                            "state": self.state,
                            "squat_count": self.squat_count,
                            "checks": {"back": None, "knee": None, "ankle": None},
                            "avg_score": self.avg_score,
                            "recent": self.recent_rep_feedback,
                            "message": message
                        }
                else:
                    self.track["slow_frames"] = 0

        elif self.state == "STAND":
            # 허리 각도 측정 및 피드백
            if BACK_USE_TRUNK_ANGLE:
                ang = compute_trunk_angle(lm, side)
                base = self.track.get("trunk_angle_baseline")
                if base is not None:
                    extra_angle = max(0.0, ang - base)
                    threshold = (TRUNK_GOOD_EXTRA + TRUNK_BAD_EXTRA) / 2.0
                    checks["back"] = "Good" if extra_angle <= threshold else "Bad"

            # 무릎 각도 체크 (충분히 펴졌는지 확인)
            knee_angle = check_knee_angle(lm, side)
            checks["knee"] = "Good" if knee_angle > 160 else "Bad"

            # 무릎-발목 정렬 체크
            knee_ankle_aligned = check_knee_ankle_alignment(lm, side)
            checks["ankle"] = "Good" if knee_ankle_aligned else "Bad"
            if inst_vel > 0 and hip_y >= knee_y - ENTER_SIT_GAP:
                self.state = "SIT"
                self.sit_start_time = time.time()
                self.track.update({
                    "hip_bottom": hip_y,
                    "knee_bottom": knee_y,
                    "sit_frames": 0,
                    "bottom_locked": False,
                    "rise_frames": 0,
                    "hip_history": [hip_y],
                    "knee_angle_min": None,
                    "back_dev_max": 0.0,
                    "slow_frames": 0
                })

        result = {
            "state": self.state,
            "squat_count": self.squat_count,
            "checks": checks,
            "avg_score": self.avg_score,
            "recent": self.recent_rep_feedback,
            "message": message if message else ""
        }
        return result