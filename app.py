import cv2
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import requests
from collections import deque
from flask import Flask, render_template, Response, request, jsonify, send_file

from handpoints_utils import create_hands, extract_two_hand_vector, extract_normalized_landmarks

# ---------------- Config ----------------
SEQ_LEN = 30
FEATURE_DIM = 126
DYNAMIC_MODEL_FILE = "gesture_lstm_pytorch.pth"
DYNAMIC_LABELS_FILE = "gestures_dynamic.txt"

ALPHABET_MODEL_FILE = "alphabet_mlp.pt"
ALPHABET_LABELS_FILE = "alphabet_letters.json"
ALPHABET_INPUT_DIM = 63

TTS_FILE = "tts_output.mp3"
SUPPORTED_TTS_LANGS = {"en","hi","ta","ml","te","kn"}

# ---------------- Flask ----------------
app = Flask(__name__)

# ---------------- Models ----------------
class GestureLSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_labels(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

gesture_labels = load_labels(DYNAMIC_LABELS_FILE)
num_classes = len(gesture_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gesture_model = GestureLSTM(FEATURE_DIM, num_classes).to(device)
gesture_model.load_state_dict(torch.load(DYNAMIC_MODEL_FILE, map_location=device))
gesture_model.eval()

# Alphabet model
class AlphaMLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class AlphabetPredictor:
    def __init__(self):
        with open(ALPHABET_LABELS_FILE) as f:
            self.labels = json.load(f)
        self.model = AlphaMLP(ALPHABET_INPUT_DIM, len(self.labels))
        self.model.load_state_dict(torch.load(ALPHABET_MODEL_FILE, map_location="cpu"))
        self.model.eval()
        self.window = deque(maxlen=7)
        self.conf_window = deque(maxlen=7)
        self.conf_threshold = 0.6

    def predict_frame(self, vec):
        with torch.no_grad():
            x = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
            probs = torch.softmax(self.model(x), dim=1)[0].numpy()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        letter = self.labels[idx]
        self.window.append(letter)
        self.conf_window.append(conf)
        if len(self.window) >= 3 and np.mean(self.conf_window) >= self.conf_threshold:
            return letter, conf
        return None, None

    def reset(self):
        self.window.clear()
        self.conf_window.clear()

alpha_pred = AlphabetPredictor()
typed_text = ""
last_alpha_accept = 0
ALPHA_COOLDOWN = 0.8

# Conversation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer_conv = AutoTokenizer.from_pretrained("conversation_model")
model_conv = AutoModelForSeq2SeqLM.from_pretrained("conversation_model")

gesture_buffer = []
last_detected = None

# Camera
cap = cv2.VideoCapture(0)
hands_dynamic = create_hands(max_num_hands=2)
hands_alpha = create_hands(max_num_hands=1)

# ---------------- Helpers ----------------
def translate_text(text, target):
    if target == "en": return text
    try:
        r = requests.get("https://translate.googleapis.com/translate_a/single",
                         params={"client":"gtx","sl":"en","tl":target,"dt":"t","q":text}, timeout=5)
        return r.json()[0][0][0]
    except:
        return text

def generate_sentence():
    if not gesture_buffer: return ""
    text = " ".join(gesture_buffer)
    inputs = tokenizer_conv(text, return_tensors="pt")
    out = model_conv.generate(**inputs, max_length=30)
    return tokenizer_conv.decode(out[0], skip_special_tokens=True)

# ---------------- Camera streaming ----------------
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame,1)
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ----- Dynamic -----
def record_sequence():
    seq=[]
    for _ in range(SEQ_LEN):
        ret, frame = cap.read()
        if not ret:
            seq.append(np.zeros(FEATURE_DIM)); continue
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_dynamic.process(rgb)
        if res.multi_hand_landmarks:
            vec = extract_two_hand_vector(res)
            if vec is not None:
                seq.append(vec); time.sleep(0.03); continue
        seq.append(np.zeros(FEATURE_DIM)); time.sleep(0.03)
    return np.array(seq,np.float32)

def run_dynamic(seq):
    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(gesture_model(x),dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return gesture_labels[idx], float(probs[idx])

@app.route("/predict", methods=["POST"])
def predict_dynamic():
    global last_detected

    start = time.perf_counter()   # ⬅ start timing

    seq = record_sequence()
    g, c = run_dynamic(seq)
    last_detected = g

    end = time.perf_counter()     # ⬅ end timing
    total_time = end - start
    fps = 1 / total_time

    print(f"Dynamic Total Time: {total_time*1000:.2f} ms | FPS: {fps:.2f}")

    return jsonify({"gesture": g, "confidence": f"{c*100:.2f}%"})


@app.route("/speak", methods=["POST"])
def speak_dynamic():
    data = request.get_json(force=True)
    text = data.get("text","").strip()
    lang = data.get("lang","en")
    if not text: return jsonify({"error":"empty"}),400
    tts_text = translate_text(text,lang)
    from gtts import gTTS
    gTTS(tts_text, lang=lang).save(TTS_FILE)
    return send_file(TTS_FILE, mimetype="audio/mpeg")

# ----- Alphabet -----
@app.route("/predict_alphabet", methods=["POST"])
def predict_alphabet():
    global typed_text, last_alpha_accept

    start = time.perf_counter()   # ⬅ start timing

    accepted = None
    for _ in range(20):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands_alpha.process(rgb)

        if res.multi_hand_landmarks:
            vec = extract_normalized_landmarks(res.multi_hand_landmarks[0])
            if vec is not None:
                letter, conf = alpha_pred.predict_frame(vec)
                if letter and time.time()-last_alpha_accept > ALPHA_COOLDOWN:
                    typed_text += letter
                    accepted = {"letter": letter, "conf": conf}
                    last_alpha_accept = time.time()
                    alpha_pred.reset()
                    break

    end = time.perf_counter()     # ⬅ end timing
    total_time = end - start
    fps = 1 / total_time

    print(f"Alphabet Total Time: {total_time*1000:.2f} ms | FPS: {fps:.2f}")

    return jsonify({"accepted": accepted, "typed_text": typed_text})


@app.route("/alphabet_space", methods=["POST"])
def alphabet_space():
    global typed_text; typed_text+=" "; return jsonify({"typed_text":typed_text})

@app.route("/alphabet_backspace", methods=["POST"])
def alphabet_backspace():
    global typed_text; typed_text=typed_text[:-1]; return jsonify({"typed_text":typed_text})

@app.route("/alphabet_clear", methods=["POST"])
def alphabet_clear():
    global typed_text; typed_text=""; return jsonify({"typed_text":typed_text})

@app.route("/alphabet_reset", methods=["POST"])
def alphabet_reset():
    alpha_pred.reset(); return jsonify({"ok":True})

@app.route("/alphabet_speak", methods=["POST"])
def alphabet_speak():
    global typed_text
    data = request.get_json(force=True,silent=True) or {}
    lang=data.get("lang","en")
    if not typed_text.strip(): return jsonify({"error":"empty"}),400
    tts_text=translate_text(typed_text.strip(),lang)
    from gtts import gTTS
    gTTS(tts_text, lang=lang).save(TTS_FILE)
    return send_file(TTS_FILE, mimetype="audio/mpeg")

# ----- Conversation -----
@app.route("/buffer/add", methods=["POST"])
def buffer_add():
    if last_detected: gesture_buffer.append(last_detected)
    return jsonify({"buffer":gesture_buffer})

@app.route("/buffer/clear", methods=["POST"])
def buffer_clear():
    if gesture_buffer: gesture_buffer.pop()
    return jsonify({"buffer":gesture_buffer})

@app.route("/buffer/generate", methods=["POST"])
def buffer_generate():
    return jsonify({"buffer":gesture_buffer,"sentence":generate_sentence()})

@app.route("/conversation/speak", methods=["POST"])
def conversation_speak():
    data=request.get_json(force=True)
    text=data.get("text","").strip()
    lang=data.get("lang","en")
    if not text: return jsonify({"error":"empty"}),400
    tts_text=translate_text(text,lang)
    from gtts import gTTS
    gTTS(tts_text, lang=lang).save(TTS_FILE)
    return send_file(TTS_FILE, mimetype="audio/mpeg")

# ---------------- Run ----------------
if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print(f"\nOpen this link in your browser: {url}\n")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)


