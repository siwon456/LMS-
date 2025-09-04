import os
import shutil
import torch
import pickle
from flask import Flask, request, redirect, url_for, render_template
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from konlpy.tag import Okt

# ─── 불용어 사전 로드 ─────────────────────────────────────────────────────────
# 프로젝트 루트에 stop_dict.txt 파일(한 줄에 하나씩 불용어) 위치
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), 'stop_dict.txt')
def load_stopwords(path=STOPWORDS_PATH):
    with open(path, encoding='utf-8') as f:
        words = [w.strip() for w in f if w.strip()]
    return set(words)

STOPWORDS = load_stopwords()
okt = Okt()

# ─── 1. 데이터 수집 & 말뭉치(Corpus) ─────────────────────────────────────────
def collect_corpus_from_video(video_path):
    """말뭉치용 텍스트 추출: 동영상→오디오→Whisper→원문 리턴"""
    audio_path = os.path.splitext(video_path)[0] + '.wav'
    clip = VideoFileClip(video_path)
    # MoviePy 버전에 따라 verbose, logger 인자 제거
    clip.audio.write_audiofile(audio_path)
    clip.close()
    result = whisper_model.transcribe(audio_path, language="ko")
    return result.get("text", "")

# ─── 2. 형태소 기반 전처리 ─────────────────────────────────────────────────────
def preprocess(text):
    # 형태소 분석
    morphs = okt.morphs(text, stem=True)
    # 소문자화, 한글/영문 필터, 불용어 제거
    clean = []
    for m in morphs:
        m_low = m.lower()
        if re.fullmatch(r"[가-힣a-z]+", m_low) and m_low not in STOPWORDS and len(m_low) > 1:
            clean.append(m_low)
    return clean

# ─── 3~7. 기존 학습/벡터/모델 코드 (필요 시 사용) ─────────────────────
def split_data(corpus_list, labels):
    return train_test_split(corpus_list, labels, test_size=0.2, random_state=42)

def vectorize(train_texts, test_texts):
    vect = TfidfVectorizer(max_features=5000)
    X_train = vect.fit_transform(train_texts)
    X_test  = vect.transform(test_texts)
    return X_train, X_test, vect

def train_model(X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

def save_model(clf, vect, path="final_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"model":clf, "vect":vect}, f)

def load_model(path="final_model.pkl"):
    data = pickle.load(open(path, "rb"))
    return data["model"], data["vect"]

def predict_text(text, clf, vect):
    tokens = preprocess(text)
    X = vect.transform([" ".join(tokens)])
    return clf.predict(X)[0]

# ─── Flask 앱 설정 & Whisper 모델 프리로드 ───────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=device)
# clf, vect = load_model("final_model.pkl")  # 필요 시 주석 해제

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(save_path)
    return redirect(url_for('show_video', filename=file.filename))

@app.route('/video/<filename>')
def show_video(filename):
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('show_video.html', filename=filename, video_files=files)

@app.route('/transcribe_and_predict/<filename>')
def transcribe_and_predict(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    raw = collect_corpus_from_video(video_path)
    tokens = preprocess(raw)
    freq = Counter(tokens)
    top10 = freq.most_common(10)
    return render_template('result.html', raw_text=raw, top10=top10)

if __name__ == '__main__':
    app.run(debug=True)
