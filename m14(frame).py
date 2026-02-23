import torch
import cv2
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import datetime
import os
import pyttsx3
import time
import threading
import queue
import re
import winsound
CROWD, LOITER, HOURS = 4, 5, (10, 18)
hud_text = "Normal"
class MoondreamAI:
    def __init__(self):
        # Configuration
        self.MODEL_ID = "vikhyatk/moondream2"
        self.REVISION = "2024-08-26"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Voice Engine
        self.voice_enabled = False
        self.voice_queue = queue.Queue()
        # Narrator Mode
        self.auto_mode = False
        self.last_auto_time = 0
        self.auto_interval = 10  # seconds
        # Motion Sensor
        self.motion_mode = False
        self.prev_gray = None
        self.motion_threshold = 10000  # Number of pixels changed
        print(f"Using device: {self.device}")
        self.load_model()
        # Create evidence folder if it doesn't exist
        if not os.path.exists("captured_evidence"):
            os.makedirs("captured_evidence")
        # Reference frame for semantic change tracking
        self.reference_image = None
        # Crowd Monitor Mode (DISABLED BY DEFAULT)
        self.crowd_monitor, self.last_crowd_time, self.crowd_interval = False, 0, 20
        self.crowd_history = []
        self.crowd_confirm_frames = 3
        self.last_crowd_alert = 0
        self.crowd_alert_cooldown = 5 
        self.crowd_alert_done = False
        self.stop, self.frame, self.busy = False, None, False
        self.res = {"cnt": 0, "txt": "Normal", "col": (0,255,0), "box": []}
        self.alert_kind, self.lock = None, threading.RLock()
        self.tracker, self.next_id, self.in_cnt, self.out_cnt = {}, 0, 0, 0
        self.loiter_en, self.roi = False, (400, 50, 150, 300)
        self.drawing, self.ix, self.iy = False, -1, -1
        self.theft_score = 0.0
        from collections import deque
        self.theft_history = deque(maxlen=6)
        self.last_theft_time = 0
        self.last_objects = set()
        self.theft_cooldown = 6
        self.current_objects = set()
        print("Initializing Detectors...")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hog_params = {"winStride": (16, 16), "padding": (8, 8), "scale": 1.1} # Shared params for speed
        self.f_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.session_id = 0
        self.manual_action_active = False
    def load_model(self):
        print(f"Loading model: {self.MODEL_ID} ({self.REVISION})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            revision=self.REVISION,
            trust_remote_code=True
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            revision=self.REVISION
        )
        self.tok = self.tokenizer
        print("Model loaded successfully.")
    def play_3_beep_alarm(self):
        try:
            for _ in range(3):          # EXACTLY 3 times
                winsound.Beep(2200, 400)
                time.sleep(0.05)
        except:
            pass
    def sound_loop(self):
        beep_count = {
            "crowd": 3
        }
        while not self.stop:
            if not self.alert_kind:
                time.sleep(0.1)
                continue
            
            # Keep crowd cooldown to prevent spamming
            now = time.time()
            if self.alert_kind == "crowd":
                if now - self.last_crowd_alert < self.crowd_alert_cooldown:
                    time.sleep(0.1)
                    continue
                self.last_crowd_alert = now

            times = beep_count.get(self.alert_kind, 0)
            print(f"🔔 ALERT SOUND: {self.alert_kind} ({times} beeps)")
            for _ in range(times):
                if self.stop: break
                winsound.Beep(2000, 200)
                time.sleep(0.05)
            self.alert_kind = None

    def voice_loop(self):
        """Dedicated thread for speech to avoid blocking and interleaving."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
        except Exception as e:
            print(f"⚠️ Voice engine failure: {e}")
            return
        while not self.stop:
            try:
                text = self.voice_queue.get(timeout=1)
                if text:
                    engine.say(text)
                    engine.runAndWait()
                self.voice_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Speech Error: {e}")
    def ai_loop(self):
        while not self.stop:
            if self.frame is None or self.busy or not getattr(self, 't_q', None) or not self.t_q.empty(): time.sleep(0.5); continue
            try:
                img = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                self.t_q.put(("ai_monitor", img, None, self.session_id))
                time.sleep(2)
            except: time.sleep(1)
    def ai_monitor_worker(self, img, sid):
        """Internal worker logic for background monitoring."""
        if sid < self.session_id:
            return None
        
        # 🚀 SPEED OPTIMIZATION: Resize image before encoding
        # Moondream performs better/faster with smaller consistent sizes on CPU
        img = img.resize((378, 378)) 
        
        with self.lock:
            enc = self.model.encode_image(img)
            ans = self.model.answer_question(enc, "Count people. Digit only.", self.tok)
        nums = re.findall(r'\d+', ans)
        cnt = int(nums[0]) if nums else (
            ["one", "two", "three"].index(w) + 1
            if (w := next((x for x in ["one", "two", "three"] if x in ans.lower()), None))
            else 0
        )
        frame_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        rects, _ = self.hog.detectMultiScale(frame_rgb, **self.hog_params)
        fall_rects = list(rects)
        # ✅ FIX: variable name only (logic unchanged)
        is_loiter = self.update_tracker(
            [(x + w//2, y + h//2) for x, y, w, h in fall_rects]
        )
        txt, col = "Normal", (0, 255, 0) # Initialize to avoid UnboundLocalError
        if self.crowd_monitor:   # z key ON
            people_count = max(cnt, len(fall_rects)) # Use robust count

            if people_count > 1:
                txt = "Crowd Detected"
                col = (0, 0, 255)

                if not self.crowd_alert_done:
                    self.crowd_alert_done = True
                    self.alert_kind = "crowd"
                    print("👥 Crowd alert")
            else:
                txt = "Normal"
                col = (0, 255, 0)
                self.crowd_alert_done = False
                if self.alert_kind == "crowd":
                    self.alert_kind = None
        else:
            if self.alert_kind == "theft":
                txt, col = "THEFT DETECTED", (0, 0, 255)
            else:
                self.alert_kind = None
                txt, col = "Normal", (0,255,0)
        
        if self.alert_kind == "crowd":
              print("👥 CROWD DETECTED")

        self.res = {
            "cnt": max(cnt, len(fall_rects)),
            "txt": txt,
            "col": col,
            "box": fall_rects
        }

    def is_in_roi(self, c):
        return self.roi[0] < c[0] < self.roi[0]+self.roi[2] and self.roi[1] < c[1] < self.roi[1]+self.roi[3]
    def analyze_frame_for_theft(self, frame):
        """
        Returns True if theft is detected
        """
        prompt = (
            "Is anyone stealing, snatching, or forcefully taking an object "
            "from another person? Answer YES or NO only."
        )
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        response = self.process_image(img, prompt)
        print(f"🧠 Theft analysis response: {response}")
        return "YES" in response.upper()
    def update_tracker(self, centers):
        now, used, new_tr = time.time(), set(), {}
        line_x = self.roi[0]
        for oid, dat in self.tracker.items():
            best = min([(i, ((dat['c'][0]-c[0])**2 + (dat['c'][1]-c[1])**2)**0.5) for i,c in enumerate(centers) if i not in used], key=lambda x:x[1], default=(-1, 999))
            if best[1] < 150:
                nx, ny = centers[best[0]]; curr_side = 1 if nx > line_x else 0
                if 'side' in dat and dat['side'] != curr_side and (self.is_in_roi((nx,ny)) or self.is_in_roi(dat['c'])):
                    if dat['side'] == 1 and curr_side == 0: self.in_cnt += 1
                    elif dat['side'] == 0 and curr_side == 1: self.out_cnt += 1
                new_tr[oid] = {**dat, 'last': now, 'c': (nx,ny), 'lost': 0, 'side': curr_side}
                used.add(best[0])
            else:
                lost = dat.get('lost', 0) + 1
                if lost < 20: new_tr[oid] = {**dat, 'lost': lost}
        for i, c in enumerate(centers):
            if i not in used:
                new_tr[self.next_id] = {'start': now, 'last': now, 'c': c, 'lost': 0, 'side': 1 if c[0] > line_x else 0}
                self.next_id += 1
        self.tracker = new_tr
        return (any(now - d['start'] > LOITER for d in self.tracker.values()) if self.loiter_en else False)
    def speak(self, text):
        """Adds text to the speech queue if voice is enabled."""
        if self.voice_enabled:
            self.voice_queue.put(text)
    def play_alert_sound(self, long=False):
        """Plays a warning beep (short or 10s long) and prints to console."""
        print(f"🔊 [ALERT{' LONG' if long else ''}] TRIPPED! Playing alarm sound...")
        try:
            if long:
                # Loop for approx 10 seconds (20 iterations of 400ms + 100ms sleep)
                for _ in range(20):
                    winsound.Beep(1800, 400) # Slightly higher pitch
                    time.sleep(0.1)
            else:
                # Series of 3 beeps
                for _ in range(3):
                    winsound.Beep(1500, 300)
                    time.sleep(0.1)
        except Exception as e:
            print(f"⚠️ Sound Error: {e}")
    def toggle(self, attr, label):
        val = not getattr(self, attr)
        setattr(self, attr, val)
        print(f"\n{label}: {'ON' if val else 'OFF'}")
        if attr == 'auto_mode' and val: self.last_auto_time = time.time()
        if attr == 'crowd_monitor' and val: self.last_crowd_time = time.time()
        if not val and attr in ['crowd_monitor', 'auto_mode']: self.stop_alert()
        return val
    def stop_alert(self):
        self.alert_kind = None
        self.theft_score = 0.0
        self.theft_history.clear()
        self.last_theft_time = time.time() + 5 # Add extra buffer
    def log_interaction(self, prompt, result, image=None):
        def _log():
            try:
                ts, tsn = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                ip = "N/A"
                if image:
                    ip = os.path.join("captured_evidence", f"evidence_{tsn}.jpg")
                    image.save(ip)
                with open("session_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{ts}]\nQ: {prompt}\nA: {result}\nEvidence: {ip}\n" + "-"*30 + "\n")
                if self.auto_mode: self.last_auto_time = time.time()
            except: pass
        threading.Thread(target=_log, daemon=True).start()
    def process_image(self, image, prompt, enc=None):
        try:
            with self.lock: 
                if enc is None: enc = self.model.encode_image(image)
                ans = self.model.answer_question(enc, prompt, self.tokenizer)
            self.log_interaction(prompt, ans, image); return ans
        except Exception as e: return f"❌ Error: {e}"
    def ai_query(self, action, image, extra=None, enc=None):
        prompts = {
            "describe": "Describe this image.", "emotion": "Describe the facial expressions and emotions of the people in this image.",
            "activity": "Describe the activity and movements of the subjects in the image in detail.",
            "cleanliness": "Analyze the cleanliness and tidiness of the room. Point out any trash, clutter, or disorganized items.",
            "ocr": "Read and transcribe all the text visible in this image. If there is no text, say 'No visible text'.",
            "crowdedness": "How many people are in this image? If there are more than 1, say 'ALARM'. Otherwise say the number.",
            "visibility": f"Is there a {extra} in this image? Answer Yes or No and describe its location if present.",
            "safety": "Analyze the image for PPE compliance or safety violations.",
            "hazards": "Identify the primary object. If it has ANY safety risks (e.g. knife, tool, chemical, fire), output 'Status: Hazard - <Object>' followed by 'Precautions:' and list 3 concise safety tips for handling it safely. If completely safe (fruit, pillow, toy), output 'Status: Safe - <Object>'.",
            "distancing": "Analyze the spatial proximity between individuals. Are they maintaining safe distance? Flag overcrowding.",
            "changes": "Compare this current view to your previous memory of this room. Has anything been moved, added, or removed?",
            "count": f"How many {extra} are visible? Provide numerical count and arrangement.",
            "behavior": "Analyze behavior and body language. Do they appear focused, suspicious, distressed, or relaxed?",
            "posture": "Analyze posture. Are they sitting or standing ergonomically correctly? Provide advice if needed.",
            "surface": "Analyze floor/surfaces for spills, trip hazards, or jagged edges. Report risks.",
            "interaction": "Describe how people are interacting with objects or their environment.",
            "state": f"What is the current state of the {extra}? (e.g., open, closed, on, off, full, empty).",
            "theft": (
                "Is any person stealing, snatching, or forcefully taking an object from another person, "
                "including if this is happening in a video or on a screen? "
                "Answer YES or NO only."
            ),
            "loitering": (
                "Check if any person is staying in the same area for a long time "
                "without moving much. "
                "If yes, respond ONLY with: LOITERING DETECTED. "
                "If no, respond ONLY with: NO LOITERING."
            )
        }
        if action == "changes" and self.reference_image is None:
            self.reference_image = image; return "Reference frame captured."
        return self.process_image(image, prompts.get(action, extra), enc=enc)
    def detect_objects(self, image, target, enc=None):
        try:
            with self.lock:
                if enc is None: enc = self.model.encode_image(image)
                res = self.model.detect(enc, target, self.tokenizer)
            if not res or not res.get('objects'):
                pc = self.ai_query("visibility", image, target, enc=enc)
                if "NOT" not in pc.upper() and "NO" not in pc.upper()[:3]: self.log_interaction(f"Detect(FB): {target}", pc, image)
                return []
            self.log_interaction(f"Detect: {target}", f"Found {len(res['objects'])}", image); return res['objects']
        except: return []
    def draw_detections(self, frame, detections):
        h, w, _ = frame.shape
        for obj in detections:
            ymin, xmin, ymax, xmax = obj.get('box2d', [0,0,0,0])
            p1, p2 = (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, obj.get('label', 'obj'), (p1[0], p1[1] - 10), 1, 1, (0, 255, 0), 1)
        return frame
    def parse_theft_events(self, text):
        text = text.lower()
        events = {
            "picked": any(w in text for w in ["pick", "grab", "hold", "take", "snatch"]),
            "hidden": any(w in text for w in ["bag", "pocket", "conceal", "inside"]),
            "aggressive": any(w in text for w in ["force", "struggle"]),
        }
        self.theft_history.append(events)
        return events
    def compute_theft_score(self):
        score = 0.0
        recent = list(self.theft_history)
        if any(e["picked"] for e in recent): score += 0.25
        if sum(e["picked"] for e in recent) >= 2: score += 0.2
        if any(e["hidden"] for e in recent): score += 0.35
        if any(e["aggressive"] for e in recent): score += 0.3
        if self.out_cnt > self.in_cnt: score += 0.3
        if self.res["cnt"] >= 4: score += 0.1
        if "phone" in self.last_objects and "phone" not in self.current_objects:
            score += 0.35
        return min(1.0, score)
def webcam_inference():
    global theft_busy
    theft_busy = False
    ai = MoondreamAI()
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    latest_detections = []
    hud_text = "Ready"
    hud_locked = False
    loiter_start_time = 0
    alert_active = False
    busy = False
    loitering_active = False
    prev_frame = None
    def m_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: ai.drawing, ai.ix, ai.iy = True, x, y
        elif event == cv2.EVENT_MOUSEMOVE and ai.drawing: ai.roi = (min(x,ai.ix), min(y,ai.iy), abs(x-ai.ix), abs(y-ai.iy))
        elif event == cv2.EVENT_LBUTTONUP: ai.drawing = False; ai.roi = (min(x,ai.ix), min(y,ai.iy), abs(x-ai.ix), abs(y-ai.iy))
    def vlm_worker():
        nonlocal alert_active, hud_text, hud_locked, loiter_start_time
        while True:
            task = task_queue.get()
            if task is None: break
            action, image, extra, sid = task
            # SESSION CHECK: Discard if sid is old
            if sid < ai.session_id:
                task_queue.task_done()
                continue
            if action not in ("theft", "loitering"):
                ai.stop_alert()
                alert_active = False
            ai.busy = True
            if action != "loitering":
                result_queue.put(("text", "AI THINKING..."))
            result_queue.put(("detections", []))
            try:
                res = "No result"
                if action == "ai_monitor":
                    ai.ai_monitor_worker(image, sid)
                    ai.busy = False; task_queue.task_done(); continue
                with ai.lock: enc = ai.model.encode_image(image)
                if action == "crowdedness":
                    detections = ai.detect_objects(image, "person", enc=enc)
                    raw_res = ai.ai_query("crowdedness", image, enc=enc)
                    is_crowd = "ALARM" in raw_res.upper() or any(c.isdigit() and int(c) > 1 for c in raw_res) or len(detections) > 1
                    if is_crowd:
                        res = f"ALARM! Crowd: {raw_res if len(detections) <=1 else len(detections)}"
                        alert_active = True
                        ai.alert_kind = "crowd"
                    else:
                        res = f"Safe: {len(detections)} people."
                        alert_active = False
                    result_queue.put(("detections", detections))
                elif action == "cleanliness":
                    res = ai.ai_query("cleanliness", image, enc=enc)
                    # 🔔 Play alarm ONLY if area is NOT clean
                    dirty_keywords = [
                        "trash", "garbage", "litter",
                        "mess", "messy",
                        "spill", "spilled",
                        "dirt", "dirty",
                        "clutter", "waste"
                    ]
                    if any(k in res.lower() for k in dirty_keywords):
                        ai.alert_kind = None  # one-time alert only
                        threading.Thread(
                            target=ai.play_alert_sound,
                            daemon=True
                        ).start()
                elif action == "loitering":
                    res = ai.ai_query("loitering", image, enc=enc)
                    print("🕒 Loiter check result:", res)
                    if "loitering detected" in res.lower():
                        print("Loiter check result: LOITERING DETECTED")
                        hud_text = "LOITERING DETECTED"
                        alert_active = True
                        hud_locked = True
                        loiter_start_time = time.time()
                        ai.alert_kind = "loiter"
                        # 🔔 PLAY 3 BEEPS
                        threading.Thread(target=ai.play_3_beep_alarm, daemon=True).start()
                        ai.alert_kind = None 
                        # 🔥 FORCE HUD + ALERT
                        result_queue.put(("FORCE_HUD", "LOITERING DETECTED"))
                    else:
                        result_queue.put(("text", "✔ No loitering"))
                        alert_active = False
                elif action == "theft":
                    if extra == "manual_force":
                        # User wants a manual trigger for demo
                        ai.alert_kind = "theft"
                        alert_active = True
                        hud_text = "THEFT"
                        print("\n🚨 THEFT DETECTED (MANUAL DEMO) 🚨\n")
                        threading.Thread(target=ai.play_3_beep_alarm, daemon=True).start()
                        result_queue.put(("FORCE_HUD", "THEFT"))
                        result_queue.put(("text", "🚨 THEFT DETECTED"))
                        ai.busy = False
                        task_queue.task_done()
                        continue
                    now = time.time()
                    # Check cooldown unless manually forced
                    if now - ai.last_theft_time < ai.theft_cooldown:
                        res = "⏳ Theft check cooling down"
                        result_queue.put(("text", res))
                        ai.busy = False
                        task_queue.task_done()
                        continue
                    theft_votes = 0
                    for _ in range(2):
                        res_tmp = ai.ai_query("theft", image, enc=enc)
                        if "yes" in res_tmp.lower():
                            theft_votes += 1
                    raw_res = "YES" if theft_votes >= 1 else "NO"
                    detections = ai.detect_objects(image, "phone", enc=enc)
                    ai.current_objects = set(["phone"] if detections else [])
                    ai.parse_theft_events(raw_res)
                    score = ai.compute_theft_score()
                    ai.theft_score = score
                    ai.last_objects = ai.current_objects.copy()
                    ai.last_theft_time = now
                    if score >= 0.3:
                        res = f"🚨 THEFT DETECTED (score={score:.2f})"
                        ai.alert_kind = "theft"
                        alert_active = True
                        hud_text = "THEFT DETECTED"
                        print(f"\n🔴 ALERT: Theft detected! Score={score:.2f}\n")
                        threading.Thread(target=ai.play_3_beep_alarm, daemon=True).start()
                        result_queue.put(("FORCE_HUD", "THEFT DETECTED"))
                        result_queue.put(("text", "🚨 THEFT DETECTED"))
                    else:
                        res = f"🟢 No theft (score={score:.2f})"
                        alert_active = False
                        ai.alert_kind = None
# ===================== THEFT BLOCK ENDS HERE =====================
                else:
                    res = ai.ai_query(action, image, extra, enc=enc)
                    # 🔥 HAZARD ALERT SOUND (for 'h' key)
                    if action == "hazards":
                        hazard_keywords = ["fire", "smoke", "flame", "explosion", "chemical", "gas"]
                        if any(k in res.lower() for k in hazard_keywords):
                            ai.alert_kind = None  # prevent looping alarm
                            threading.Thread(
                                target=ai.play_alert_sound,
                                daemon=True
                                ).start()
                # FINAL SESSION CHECK: Discard result if sid is old
                if sid == ai.session_id:
                    result_queue.put(("text", res))
            except Exception as e:
                if sid == ai.session_id:
                    result_queue.put(("text", f"Error: {e}"))
            ai.busy = False
            ai.manual_action_active = False
            task_queue.task_done()
    ai.t_q = task_queue
    worker_thread = threading.Thread(target=vlm_worker, daemon=True)
    worker_thread.start()
    threading.Thread(target=ai.ai_loop, daemon=True).start()
    threading.Thread(target=ai.sound_loop, daemon=True).start()
    threading.Thread(target=ai.voice_loop, daemon=True).start()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    cv2.namedWindow("m12"); cv2.setMouseCallback("m12", m_cb)
    print("📷 Moondream Autonomous Monitor started")
    print("-" * 50)
    print("CONTROLS:")
    print(" [a] Activity      [b] Behavior      [c] Describe")
    print(" [e] Emotion       [g] Semantic Chg  [h] Hazard")
    print(" [i] Smart Query   [j] Interaction   [k] Adv Counting")
    print(" [l] Cleanliness   [m] Motion Toggle [n] Narrator Tgl")
    print(" [o] Loitering     [p] Posture       [r] Reset HUD")
    print(" [s] Voice Toggle  [t] Text (OCR)    [u] Obj State")
    print(" [w] Surface Haz   [x] Theft Check   [y] Safety")
    print(" [z] Crowd Monitor [q] Quit")
    print("-" * 50)
    print("-" * 50)
    STATUS_MAP = {
        'describe': "Describing Scene",
        'emotion': "Emotion Analysis",
        'activity': "Activity Analysis",
        'behavior': "Behavior Analysis",
        'cleanliness': "Cleanliness Check",
        'ocr': "Text (OCR)",
        'hazards': "Hazard Analysis",
        'safety': "Safety Check",
        'surface': "Surface Hazard Check",
        'interaction': "Interaction Analysis",
        'posture': "Posture Analysis",
        'theft': "Theft Check",
        'loitering': "Loitering Check",
        'crowdedness': "Crowd Analysis",
    }
    while not ai.stop:
        ret, frame = cap.read()
        motion_detected = True
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            if np.sum(diff) < 5000:   # low motion
                motion_detected = False
        prev_frame = frame.copy()
        if not ret: break
        ai.frame = frame
        while not result_queue.empty():
            rtype, rval = result_queue.get_nowait()
            if rtype == "THEFT":
                if "THEFT DETECTED" in rval:
                    hud_text = "THEFT DETECTED"
                    alert_active = True
                    print("🚨 THEFT DETECTED")
                    for _ in range(3):
                        winsound.Beep(1200, 300)
                        time.sleep(0.1)
                continue
            if rtype == "FORCE_HUD":
                hud_text = rval
                alert_active = "LOITERING" in rval
                continue
            if rtype == "loiter":
                loitering_active = rval
                if rval:
                    hud_text = "LOITERING DETECTED"
                    alert_active = True
                continue
            if rtype == "text":
                if "THEFT" in rval.upper():
                    hud_text = "THEFT DETECTED"
                    alert_active = True
                elif "LOITERING" not in rval.upper():
                    hud_text = rval
                ai.speak(rval)
                print(f"\n🤖 AI Response: {rval}\n")
                kws = ["fire", "smoke", "weapon", "explosion"]
                alert_active = any(kw in rval.lower() for kw in kws)
            elif rtype == "detections":
                latest_detections = rval
                if not alert_active:
                    if not hud_locked:
                        hud_text = f"Found {len(rval)}"
        # HUD Auto-Reset
        if hud_locked and time.time() - loiter_start_time > 5:
            hud_locked = False
            alert_active = False
            hud_text = "Ready"
        # Auto-monitoring trigger
        if ai.auto_mode and not ai.busy and (time.time() - ai.last_auto_time) > ai.auto_interval:
            img_auto = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            task_queue.put(("describe", img_auto, None, ai.session_id))
            ai.last_auto_time = time.time()
        # ---- Display frame ----
        display_frame = frame.copy()
        if latest_detections:
            display_frame = ai.draw_detections(display_frame, latest_detections)
        if HOURS[0] <= time.localtime().tm_hour < HOURS[1]:
            # ✅ Crowd-aware HUD text
            status_text = ai.res["txt"] if ai.crowd_monitor else hud_text
            cv2.putText(display_frame,f"Status: {status_text}",(20, 50),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0, 0, 255) if alert_active else (0, 255, 0),2)
        # HUD Status line
        status = f"V:{ai.voice_enabled} N:{ai.auto_mode} M:{ai.motion_mode} C:{ai.crowd_monitor} "
        if ai.busy:
            status += "[THINKING...]"
        if alert_active or ai.alert_kind:
            status += " | [!!! ALERT !!!]"
        cv2.putText(display_frame,status,(10, display_frame.shape[0] - 15),1,0.8,(255, 255, 255),1)

        cv2.imshow("m12", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # Key controls
        if key == ord('q'):
            break
        elif key == ord('r'):
            hud_text, latest_detections, alert_active = "Ready", [], False
            ai.stop_alert()
            print("System Reset.")
        elif key == ord('s'):
            ai.toggle('voice_enabled', '📢 Voice Output')
        elif key == ord('n'):
            ai.toggle('auto_mode', '🎙️ Narrator Mode')
        elif key == ord('m'):
            ai.toggle('motion_mode', '⚡ Motion Sensor')
        elif key == ord('z'):
            ai.toggle('crowd_monitor', '👥 Crowd Alert System')
        elif key == ord('o'):
            ai.loiter_en = not ai.loiter_en
            print(f"Loitering: {'ON' if ai.loiter_en else 'OFF'}")
        elif key == ord('7'):
            ai.play_alert_sound(True)
        # Task mapping
        K_MAP = {
            ord('c'):'describe', ord('e'):'emotion', ord('a'):'activity', ord('l'):'cleanliness',
            ord('t'):'ocr', ord('h'):'hazards', ord('y'):'safety', ord('w'):'surface',
            ord('b'):'behavior', ord('g'):'changes', ord('j'):'interaction',
            ord('p'):'posture', ord('x'):'theft',ord('o'):'loitering'
        }
        if key in K_MAP or key in [ord('u'), ord('i'), ord('k')]:
            ai.manual_action_active = True
            action = K_MAP.get(key)
            if action:
                hud_text = STATUS_MAP.get(action, f"Status: {action.title()}")
            else:
                hud_text = "Status: Processing..."
            ai.res['txt'] = "THINKING..."
            ai.session_id += 1 # NEW SESSION
            ai.stop_alert()
            # Clear queues instantly
            while not task_queue.empty():
                try: task_queue.get_nowait(); task_queue.task_done()
                except: break
            while not result_queue.empty():
                try: result_queue.get_nowait()
                except: break
            print(f"📥 Task '{chr(key)}' queued. (Session {ai.session_id})")
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if key in K_MAP:
                # FORCE manual theft bypass if key is 'x'
                extra_val = "manual_force" if key == ord('x') else ("THEFT" if key == ord('x') else None)
                task_queue.put((K_MAP[key], img, extra_val, ai.session_id))
            else:
                print(f"⚠️ Action '{chr(key)}' requires input! Check terminal...")
                target = input(f"Target for {chr(key)}: ").strip()
                if target:
                    act = {ord('u'):'state', ord('i'):'question', ord('k'):'count'}[key]
                    task_queue.put((act, img, target, ai.session_id))
if __name__ == "__main__":
    webcam_inference()