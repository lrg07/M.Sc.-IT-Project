import sys
print(sys.executable)
print(sys.version)

import speech_recognition as sr
import random
import requests
import webbrowser
import datetime
import pyautogui
import wikipedia
import pywhatkit as pwk
import time
from gtts import gTTS
import pygame
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from twilio.rest import Client
from plyer import notification
import re
import sqlite3
import threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import librosa
import soundfile as sf
from scapy.all import sniff, IP, TCP
import spacy
from spacy.training import Example
import pandas as pd
import hashlib
import psutil
import tkinter as tk
from tkinter import scrolledtext
from bs4 import BeautifulSoup  # NEW: for Google News RSS [web:52]

# =====================================================================
# WEATHER (Open-Meteo API, no key) [web:39]
# =====================================================================
def get_weather(request_text="Nagpur"):
    try:
        city = "Nagpur"
        text_lower = request_text.lower()

        city_match = re.search(r'(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\s|$|today|tomorrow)', text_lower)
        if city_match:
            city = city_match.group(1).strip()
        else:
            m2 = re.search(r'\b([a-zA-Z]{4,})\b.*\bweather\b', text_lower)
            if m2:
                city = m2.group(1).strip().title()

        print(f"[DEBUG] Weather city detected: {city}")

        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_resp = requests.get(geo_url, params={"name": city, "count": 1}, timeout=10)
        geo_data = geo_resp.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find weather data for {city}."

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        city_name = geo_data["results"][0]["name"]

        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "temperature_unit": "celsius",
            "windspeed_unit": "kmh",
            "timezone": "Asia/Kolkata"
        }
        w_resp = requests.get(weather_url, params=params, timeout=10)
        w_data = w_resp.json()
        current = w_data.get("current_weather", {})

        temp = current.get("temperature", 0)
        wind = current.get("windspeed", 0)
        code = current.get("weathercode", 0)

        weather_desc = {
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "moderate drizzle",
            55: "dense drizzle", 61: "slight rain", 63: "moderate rain", 65: "heavy rain",
            71: "slight snow", 73: "moderate snow", 75: "heavy snow", 95: "thunderstorm"
        }.get(code, "pleasant conditions")

        return f"Weather in {city_name}: {temp} degrees Celsius, {weather_desc}, winds {wind} kilometers per hour."
    except Exception as e:
        print("Weather error:", e)
        return "Weather service temporarily unavailable."

# =====================================================================
# NEWS (Google News RSS – NO API KEY) [web:48]
# =====================================================================
def get_news():
    try:
        rss_url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(rss_url, timeout=10)
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")[:3]

        if not items:
            return "No headlines available right now."

        headlines = []
        for item in items:
            title = item.title.text.strip()
            headlines.append(title)

        speak_text = "Top India headlines: "
        for i, h in enumerate(headlines, 1):
            speak_text += f"Headline {i}: {h[:70]}... "

        print("[DEBUG] News prepared:", speak_text[:120])
        return speak_text
    except Exception as e:
        print("News error:", e)
        return "News service temporarily unavailable."

# =====================================================================
#Calculator
#=======================================================================
def open_calculator():
    try:
        pyautogui.hotkey("win")
        time.sleep(0.7)
        pyautogui.typewrite("calculator")
        time.sleep(0.7)
        pyautogui.press("enter")
        time.sleep(1.5)
        return True
    except:
        return False

def parse_spoken_calculation(text):
    t = text.lower()
    t = re.sub(r'\brogers\b', '', t)
    t = t.replace("plus","+").replace("add","+").replace("minus","-")
    t = t.replace("times","*").replace("into","*").replace("divide by","/")
    allowed = "0123456789+-*/."
    expr = "".join(c for c in t if c in allowed)
    return expr if len(expr)>2 else None

def calc_type_expression(expr):
    for ch in expr:
        if ch in "0123456789+-*/.":
            pyautogui.press(ch)
            time.sleep(0.1)
    pyautogui.press("enter")


# =====================================================================
# SPEAK FUNCTION
# =====================================================================
def speak(text, emotion='neutral'):
    try:
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "voice.mp3"
        tts.save(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.2)
        pygame.mixer.quit()
        os.remove(filename)
    except Exception as e:
        print("TTS Error:", e)

# =====================================================================
# 1. INTENT CLASSIFIER (with NEWS & WEATHER)
# =====================================================================
class IntentClassifier:
    def __init__(self):
        self.spacy_nlp = None
        self.tfidf_model = None
        self.lr_model = None
        self.load_models()

    def load_models(self):
        try:
            self.spacy_nlp = spacy.load("intent_model")
        except:
            print("spaCy model not found, using TF-IDF backup")
            self.spacy_nlp = None
            self.tfidf_model = joblib.load('tfidf_intent.pkl') if os.path.exists('tfidf_intent.pkl') else None
        try:
            self.lr_model = joblib.load('intent_classifier.pkl')
        except:
            self.train_tfidf_model()

    def train_tfidf_model(self):
        train_data = [
            ("play music", "PLAY_MUSIC"), ("play songs", "PLAY_MUSIC"),
            ("music please", "PLAY_MUSIC"),
            #calculator
            ("open calculator", "OPEN_CALC"),
            ("calculate", "OPEN_CALC"),
            #todo list
            ("add work buy milk", "ADD_TODO"),
            ("new work meeting", "ADD_TODO"),
            ("todo groceries", "ADD_TODO"),
            ("remove work milk", "REMOVE_TODO"),
            ("delete meeting", "REMOVE_TODO"),
            #Time,Date
            ("what time", "GET_TIME"), ("current time", "GET_TIME"),
            ("today date", "GET_DATE"), ("what date", "GET_DATE"),
            #Youtube
            ("open youtube", "OPEN_YOUTUBE"),
            ("open notepad", "OPEN_APP"),
            ("wikipedia python", "WIKIPEDIA"),
            ("search google ai", "GOOGLE_SEARCH"),
            ("whatsapp message", "WHATSAPP"),
            ("thanks rogers", "THANKS"),
            ("show work", "SHOW_TODO"),
            ("security check", "SECURITY_CHECK"),
            ("close notepad", "CLOSE_APP"),
            # weather
            ("weather", "WEATHER"),
            ("weather in nagpur", "WEATHER"),
            ("temperature", "WEATHER"),
            ("forecast", "WEATHER"),
            # news
            ("news", "NEWS"),
            ("headlines", "NEWS"),
            ("latest news", "NEWS"),
            ("breaking news", "NEWS"),
            #quit/close
            ("quit rogers", "QUIT"),
            ("exit rogers", "QUIT"),
            ("close yourself", "QUIT"),
            ("stop listening", "QUIT"),
            ("shutdown", "QUIT"),

        ]
        df = pd.DataFrame(train_data, columns=['text', 'intent'])
        self.tfidf_model = TfidfVectorizer(max_features=1000, stop_words='english')
        X = self.tfidf_model.fit_transform(df['text'])
        y = df['intent']
        self.lr_model = LogisticRegression()
        self.lr_model.fit(X, y)
        joblib.dump(self.tfidf_model, 'tfidf_intent.pkl')
        joblib.dump(self.lr_model, 'intent_classifier.pkl')
        print("TF-IDF Intent model trained (with news & weather).")

    def predict(self, text):
        text = self.preprocess(text)
        if self.spacy_nlp:
            doc = self.spacy_nlp(text)
            intent = max(doc.cats, key=doc.cats.get)
            return intent, {"confidence": doc.cats[intent]}
        if self.tfidf_model and self.lr_model:
            X = self.tfidf_model.transform([text])
            intent = self.lr_model.predict(X)[0]
            confidence = self.lr_model.predict_proba(X).max()
            return intent, {"confidence": confidence}
        return "UNKNOWN", {"confidence": 0.0}

    @staticmethod
    def preprocess(text):
        text = text.lower().strip()
        text = re.sub(r'\brogers\b', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

intent_classifier = IntentClassifier()

# =====================================================================
# 2. CONTEXT MEMORY
# =====================================================================
class ContextMemory:
    def __init__(self):
        self.conn = sqlite3.connect('context.db', check_same_thread=False)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS sessions 
                            (session_id TEXT, timestamp TEXT, intent TEXT, entities TEXT, emotion TEXT)''')
        self.conn.commit()

    def get_session_id(self):
        mac = ':'.join(['{:02x}'.format(random.randint(0, 255)) for _ in range(6)])
        return hashlib.md5(mac.encode()).hexdigest()[:8]

    def save_context(self, session_id, intent, entities, emotion="neutral"):
        self.conn.execute(
            "INSERT INTO sessions VALUES(?,?,?,?,?)",
            (session_id, datetime.datetime.now().isoformat(), intent, str(entities), emotion)
        )
        self.conn.commit()

    def get_recent_context(self, session_id, limit=3):
        cursor = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        )
        return cursor.fetchall()

context_memory = ContextMemory()

# =====================================================================
# 3. EMOTION DETECTOR (stub)
# =====================================================================
class EmotionDetector:
    def __init__(self):
        try:
            self.model = joblib.load('emotion_model.pkl')
        except:
            self.model = None
            self.train_emotion_model()

    def train_emotion_model(self):
        print("Emotion model needs real training data")
        self.model = None

    def detect_from_audio(self, audio_path):
        if not self.model or not os.path.exists(audio_path):
            return 'neutral'
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            emotion = self.model.predict([mfccs])[0]
            return emotion
        except:
            return 'neutral'

emotion_detector = EmotionDetector()

# =====================================================================
# 4. NETWORK MONITOR
# =====================================================================
class NetworkMonitor:
    def __init__(self):
        self.suspicious_ips = set()
        self.monitoring = False

    def packet_callback(self, packet):
        if packet.haslayer(IP) and not packet.haslayer(TCP):
            src_ip = packet[IP].src
            if not src_ip.startswith('192.168.') and not src_ip.startswith('10.'):
                self.suspicious_ips.add(src_ip)

    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._sniff_packets, daemon=True)
            self.monitor_thread.start()

    def _sniff_packets(self):
        sniff(prn=self.packet_callback, store=0, stop_filter=lambda x: not self.monitoring)

    def security_report(self):
        threats = len(self.suspicious_ips)
        if threats > 0:
            return f"Alert: {threats} suspicious IPs detected."
        return "Network secure."

network_monitor = NetworkMonitor()
network_monitor.start_monitoring()

# =====================================================================
# BASIC FUNCTIONS
# =====================================================================
def WishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning, Sir! Rogers with weather, news, and security online.")
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon, Sir! Weather, news, and AI features ready.")
    else:
        speak("Good Evening, Sir! Full capabilities online including weather and news.")

def command():
    while True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            try:
                r.adjust_for_ambient_noise(source, duration=0.8)
                audio = r.listen(source, timeout=5, phrase_time_limit=6)
            except Exception as e:
                print("Listen error:", e)
                continue
        try:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
        except:
            pass
        try:
            content = r.recognize_google(audio, language="en-in")
            print("You said:", content)
            return content
        except Exception as e:
            print("STT error:", e)

# =====================================================================
# RULE OVERRIDES
# =====================================================================
def override_intent_rules(text, ml_intent, ml_info):
    t = text.lower().strip()

    if any(w in t for w in ["weather", "temperature", "climate", "forecast"]):
        return "WEATHER", ml_info
    if any(w in t for w in ["news", "headlines", "breaking news"]):
        return "NEWS", ml_info
    if any(w in t for w in ["quit rogers", "exit rogers", "shutdown", "stop listening"]):
        return "QUIT", ml_info
    if "search google" in t or ("search" in t and "google" in t):
        return "GOOGLE_SEARCH", ml_info
    if any(w in t for w in ["open calculator", "calculator", "calculate"]):
        return "OPEN_CALC", ml_info
    if t.startswith("open "):
        return "OPEN_APP", ml_info
    if t.startswith("close "):
        return "CLOSE_APP", ml_info
    if "play music" in t or "play song" in t or "songs" in t:
        return "PLAY_MUSIC", ml_info
    if "security check" in t or "network security" in t:
        return "SECURITY_CHECK", ml_info
    if "show work" in t or "show tasks" in t or "today's schedule" in t:
        return "SHOW_TODO", ml_info
    if "open youtube" in t or "youtube" in t:
        return "OPEN_YOUTUBE", ml_info
    if "what time" in t or "current time" in t:
        return "GET_TIME", ml_info
    if "date" in t and "today" in t:
        return "GET_DATE", ml_info

    if ml_info.get("confidence", 0) >= 0.6:
        return ml_intent, ml_info
    return "UNKNOWN", ml_info

# =====================================================================
# VOICE LOOP
# =====================================================================
def voice_loop_autostart(status_label, log_box, security_label):
    session_id = context_memory.get_session_id()
    while True:
        status_label.config(text="Status: Listening...")
        request = command()
        if not request:
            time.sleep(0.5)
            continue
        status_label.config(text="Status: Processing...")

        emotion = emotion_detector.detect_from_audio("temp_audio.wav")
        ml_intent, entities = intent_classifier.predict(request)
        intent, entities = override_intent_rules(request, ml_intent, entities)

        log_box.insert(tk.END, f"You said: {request}\n")
        log_box.insert(
            tk.END,
            f"Intent: {intent} | Emotion: {emotion} | Confidence: {entities.get('confidence', 0):.2f}\n\n"
        )
        log_box.see(tk.END)

        context_memory.save_context(session_id, intent, entities, emotion)
        security_label.config(text="Security: " + network_monitor.security_report())

        if intent == "NEWS":
            speak("Getting latest headlines from Google News.", emotion)
            news_text = get_news()
            speak(news_text, emotion)

        elif intent == "WEATHER":
            speak("Checking weather.", emotion)
            weather_text = get_weather(request)
            speak(weather_text, emotion)

        elif intent == "PLAY_MUSIC":
            speak("Playing your favorite tracks...", emotion)
            time.sleep(0.7)
            song = random.randint(1, 5)
            songs = [
                "https://youtu.be/suC_Y2eZtAw?list=RDsuC_Y2eZtAw",
                "https://www.youtube.com/watch?v=LK7-_dgAVQE",
                "https://www.youtube.com/watch?v=9KCtZ9r4OAw",
                "https://www.youtube.com/watch?v=pxZTbeobtW0",
                "https://www.youtube.com/watch?v=0bxVTq_A9Ss"
            ]
            webbrowser.open(songs[song-1])

        elif intent == "GET_DATE":
            date_today = datetime.datetime.now().strftime("%d/%m/%Y")
            speak(f"Today's date is {date_today}", emotion)

        elif intent == "GET_TIME":
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            speak(f"Current time is {current_time}", emotion)

        elif intent == "ADD_TODO":
            task = re.sub(r'(add task|new task|todo|add work)', '', request.lower()).strip()
            if task:
                speak(f"Task added: {task}", emotion)
                with open("todo.txt", 'a', encoding='utf-8') as f:
                    f.write(f"{task}\n")
            else:
                speak("Please specify the task sir.", 'neutral')

        elif intent == "REMOVE_TODO":
            speak("Task removal feature active.", emotion)

        elif intent == "SHOW_TODO":
            try:
                with open("todo.txt", 'r', encoding='utf-8') as f:
                    tasks = f.read().strip()
                if tasks:
                    speak("Your tasks are: " + tasks[:100], emotion)
                    notification.notify(title="Today's Tasks", message=tasks)
                else:
                    speak("No tasks pending sir.", emotion)
            except:
                speak("Todo list empty.", emotion)

        elif intent == "SECURITY_CHECK":
            report = network_monitor.security_report()
            speak(report, 'neutral')

        elif intent == "GOOGLE_SEARCH":
            query = request.lower()
            query = query.replace("search google", "").replace("google", "").replace("about", "").strip()
            if not query:
                query = "latest news"
            speak("Searching Google for " + query)
            webbrowser.open("https://www.google.com/search?q=" + query)

        elif intent == "OPEN_YOUTUBE":
            webbrowser.open("www.youtube.com")
            speak("YouTube opened sir.", emotion)

        elif intent == "OPEN_APP":
            text = request.lower()
            for w in ["open", "please", "rogers"]:
                text = text.replace(w, "")
            app = text.strip()
            if not app:
                speak("Please tell me which application to open, sir.", emotion)
            else:
                speak(f"Opening {app}...", emotion)
                time.sleep(0.7)
                try:
                    pyautogui.hotkey("win")
                    time.sleep(0.8)
                    pyautogui.typewrite(app)
                    time.sleep(0.8)
                    pyautogui.press("enter")
                except Exception as e:
                    print("OPEN_APP error:", e)
                    speak("Sorry sir, I could not open the application.", emotion)

        elif intent == "CLOSE_APP":
            text = request.lower()
            for w in ["close", "please", "rogers"]:
                text = text.replace(w, "")
            app = text.strip()
            if not app:
                speak("Please tell me which application to close, sir.", emotion)
            else:
                speak(f"Trying to close {app}...", emotion)
                alias_map = {
                    "microsoft word": "winword.exe",
                    "word": "winword.exe",
                    "google chrome": "chrome.exe",
                    "edge": "msedge.exe",
                    "calculator": "calculator.exe"
                }
                target = alias_map.get(app, None)
                closed_any = False
                if target:
                    for proc in psutil.process_iter(['name']):
                        try:
                            if proc.info['name'] and proc.info['name'].lower() == target.lower():
                                proc.terminate()
                                closed_any = True
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                else:
                    for proc in psutil.process_iter(['name']):
                        try:
                            name = (proc.info['name'] or "").lower()
                            if app in name:
                                proc.terminate()
                                closed_any = True
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                if closed_any:
                    speak(f"{app} closed successfully.", emotion)
                else:
                    speak(f"I could not find {app} running, sir.", emotion)

        elif intent == "WHATSAPP":
            speak("Sending WhatsApp message...", emotion)
            pwk.sendwhatmsg(
                "+918007537381",
                f"Hello from AI Rogers! [{datetime.datetime.now().strftime('%H:%M')}]",
                19, 42, 30, True, 10
            )

        elif intent == "OPEN_CALC":
            speak("Opening calculator.", emotion)
            if open_calculator():
                expr = parse_spoken_calculation(request)
                if expr:
                    speak(f"Calculating {expr}", emotion)
                    calc_type_expression(expr)
                else:
                    speak("Calculator ready.", emotion)
            else:
                speak("Could not open calculator.", emotion)


        elif intent == "QUIT":
            speak("Shutting down. Goodbye sir.", emotion)
            try:
                security_label.config(text="Security: Stopped")
            except:
                pass
            # stop network monitor loop
            network_monitor.monitoring = False
            # close Tkinter window and exit
            if tk._default_root is not None:
                tk._default_root.destroy()
            os._exit(0)


        else:
            speak("Command understood but not implemented yet.", emotion)

        status_label.config(text="Status: Idle")
        time.sleep(0.3)

# =====================================================================
# TKINTER HUD
# =====================================================================
root = None
def start_hud_ui():
    global root
    root = tk.Tk()
    root.title("Rogers HUD - Weather + News + Security")
    root.geometry("800x500")
    root.configure(bg="#101018")

    title_label = tk.Label(
        root, text="Rogers · Weather + News + AI HUD",
        font=("Segoe UI", 16, "bold"), bg="#101018", fg="#00ffcc"
    )
    title_label.pack(pady=10)

    status_label = tk.Label(root, text="Status: Initializing...", font=("Segoe UI", 10),
                            bg="#101018", fg="#ffffff")
    status_label.pack(pady=2)

    security_label = tk.Label(root, text="Security: Scanning...", font=("Segoe UI", 10),
                              bg="#101018", fg="#ffcc00")
    security_label.pack(pady=2)

    log_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10),
                                        bg="#181820", fg="#e0e0e0")
    log_box.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    t = threading.Thread(
        target=voice_loop_autostart,
        args=(status_label, log_box, security_label),
        daemon=True
    )
    t.start()

    root.mainloop()
#==================================================================================
#Password verification
#==================================================================================

def verify_voice_password(max_attempts=3):
    r = sr.Recognizer()
    PASSWORD_TEXT = "rogers unlock the machine"   # you will say this sentence

    for attempt in range(1, max_attempts + 1):
        print(f"Voice password attempt {attempt}/{max_attempts}")
        speak("Please say your voice password.", 'neutral')

        with sr.Microphone() as source:
            try:
                r.adjust_for_ambient_noise(source, duration=0.8)
                audio = r.listen(source, timeout=5, phrase_time_limit=4)
            except Exception as e:
                print("Listen error in verify:", e)
                continue

        try:
            spoken = r.recognize_google(audio, language="en-in").lower().strip()
            print("You said (password):", spoken)
        except Exception as e:
            print("Password STT error:", e)
            spoken = ""

        if spoken == PASSWORD_TEXT:
            speak("Voice password accepted. Welcome sir.", 'happy')
            return True
        else:
            speak("Password incorrect.", 'neutral')

    speak("Too many failed attempts. Shutting down.", 'neutral')
    return False

# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    if verify_voice_password():
        WishMe()
        start_hud_ui()
    else:
        # do not start assistant if password fails
        os._exit(0)
