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
            "trunk_angle_peak": None
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

        if self.track["hip_init"] is None:
            if self.initial_hip_y[side] is None:
                self.initial_hip_y[side] = hip_y
            self.track["hip_init"] = self.initial_hip_y[side]
        hip_init = self.track["hip_init"]

        visible_ok, vis_map = self._joints_visible(lm, side)
        if not visible_ok:
            return { "state": self.state, "squat_count": self.squat_count, "feedback": "Low visibility", "vis": vis_map }

        hh = self.track["hip_history"]
        hh.append(hip_y)
        if len(hh) > VEL_WINDOW: hh.pop(0)
        inst_vel = hip_y - hh[-2] if len(hh) >= 2 else 0.0

        back_straightness = abs(hip_y - shoulder_y)
        feedback = ""
        extra = {}

        if self.state == "START":
            feedback = "Check posture"
            if abs(hip_y - knee_y) < ENTER_SIT_GAP:
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
                    "back_dev_max": 0.0
                })
                if BACK_USE_TRUNK_ANGLE and self.track["trunk_angle_baseline"] is None:
                    ang0 = compute_trunk_angle(lm, side)
                    self.track["trunk_angle_baseline"] = ang0
                    self.track["trunk_angle_ema"] = ang0
                    self.track["trunk_angle_peak"] = ang0

        elif self.state == "SIT":
            hip = [lm[ids["HIP"]].x, hip_y]
            knee = [lm[ids["KNEE"]].x, knee_y]
            ankle = [lm[ids["ANKLE"]].x, lm[ids["ANKLE"]].y]
            knee_angle = calculate_angle(hip, knee, ankle)
            if self.track["knee_angle_min"] is None or knee_angle < self.track["knee_angle_min"]:
                self.track["knee_angle_min"] = knee_angle
            if BACK_USE_TRUNK_ANGLE:
                ang = compute_trunk_angle(lm, side)
                ema = self.track["trunk_angle_ema"]
                self.track["trunk_angle_ema"] = ang if ema is None else (1-TRUNK_EMA_ALPHA)*ema + TRUNK_EMA_ALPHA*ang
                if self.track["trunk_angle_peak"] is None or ang > self.track["trunk_angle_peak"]:
                    self.track["trunk_angle_peak"] = ang

            if hip_y > self.track["hip_bottom"]:
                self.track["hip_bottom"] = hip_y
                self.track["knee_bottom"] = knee_y
            self.track["sit_frames"] += 1

            depth = self.track["hip_bottom"] - hip_init
            if (not self.track["bottom_locked"] and
                self.track["sit_frames"] >= BOTTOM_MIN_FRAMES and
                depth >= DEPTH_MIN):
                self.track["bottom_locked"] = True
                extra["bottom"] = "OK"

            if self.track["bottom_locked"] and inst_vel < UP_VEL_THRESH:
                self.state = "RISING"
                self.track["rise_frames"] = 0
                self.track["hip_history"] = [hip_y]
                self.track["rising_start_hip"] = self.track["hip_bottom"]
                feedback = "Up..."

        elif self.state == "RISING":
            start_hip = self.track.get("rising_start_hip", self.track.get("hip_bottom", hip_y))
            target_y = min(hip_init + STAND_HIP_MARGIN, knee_y - KNEE_GAP)
            total_needed = start_hip - target_y
            if total_needed < 1e-6: total_needed = 1e-6
            rising_disp = start_hip - hip_y
            progress = clamp(rising_disp / total_needed, 0.0, 1.2)
            extra["progress"] = round(progress,2)

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
                    "trunk_angle_peak": self.track.get("trunk_angle_ema")
                })
                return {
                    "state": self.state,
                    "squat_count": self.squat_count,
                    "feedback": f"Rep {self.squat_count}",
                    "score": score,
                    "grade": grade,
                    "breakdown": bd,
                    "avg_score": self.avg_score,
                    "recent": self.recent_rep_feedback,
                    "progress": 1.0
                }
            feedback = "Rising..."

        elif self.state == "STAND":
            feedback = "Stand"
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
                    "knee_angle_min": None
                })

        return {
            "state": self.state,
            "squat_count": self.squat_count,
            "feedback": feedback,
            "avg_score": self.avg_score,
            "recent": self.recent_rep_feedback,
            **({"extra": extra} if extra else {})
        }