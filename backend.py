import os
import queue
import sys
import json
import wave
from xml.parsers.expat import model
import sounddevice as sd
import vosk
import requests
from tqdm import tqdm
from zipfile import ZipFile
from groq import Groq
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- 1. Automatic Model Downloader ---
def check_and_download_model(model_name="vosk-model-small-en-us-0.15"):
    if not os.path.exists(model_name):
        print(f"Model '{model_name}' not found. Downloading...")
        url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(f"{model_name}.zip", "wb") as file, tqdm(
            desc=model_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        
        print("Unzipping model...")
        with ZipFile(f"{model_name}.zip", 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(f"{model_name}.zip")
        print("Model ready.")
    return model_name

# --- 2. Audio Recorder ---
class AudioRecorder:
    def __init__(self, output_filename="temp_meeting.wav", sample_rate=16000):
        self.output_filename = output_filename
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.recording = False
        self.wav_file = None
        self.stream = None

    def _callback(self, indata, frames, time, status):
        if status:
            print(f"Audio Error: {status}", file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    def start(self):
        self.recording = True
        self.wav_file = wave.open(self.output_filename, "wb")
        self.wav_file.setnchannels(1)
        self.wav_file.setsampwidth(2)
        self.wav_file.setframerate(self.sample_rate)
        
        # Determine device (defaults to system default)
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=4000,
            dtype="int16",
            channels=1,
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.wav_file:
            self.wav_file.close()

    def process_queue(self):
        """Reads queue and yields data for live processing."""
        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            if self.wav_file:
                self.wav_file.writeframes(data)
            yield data

# --- 3. Speech to Text Engine ---
# --- 3. Speech to Text Engine (Groq Whisper) ---
class STTEngine:
    def __init__(self, api_key=None):
        self.client = Groq(api_key=api_key) if api_key else None

    def process_chunk(self, data):
        # We don't do live processing with Whisper (it processes files)
        # So we return None here to keep the loop running without crashing
        return None

    def transcribe_file(self, filename):
        """Sends the full audio file to Groq Whisper for accurate transcription."""
        if not self.client:
            return "Error: Groq API Key missing."
        
        try:
            with open(filename, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="whisper-large-v3", # Using the powerful Whisper model
                    response_format="json",
                    language="en",
                    temperature=0.0
                )
            return transcription.text
        except Exception as e:
            return f"Transcription Error: {e}"

# --- 4. Summarizer (Groq) ---
# --- 4. Summarizer (Groq) ---
class MeetingSummarizer:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key) if api_key else None

    def generate_summary(self, transcript):
        if not self.client:
            return "Error: Groq API Key missing."
        
        if not transcript or len(transcript) < 5:
            return "Transcript too short to summarize."

        prompt = f"""
        You are an expert secretary. Summarize this meeting transcript properly.
        
        Transcript:
        {transcript}
        
        Output format:
        ## Meeting Summary
        * **Topic:** [Topic]
        * **Key Points:** [Bulleted list]
        * **Action Items:** [List of tasks]
        """
        
        try:
            chat = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                # UPDATED MODEL NAME BELOW:
                model="llama-3.3-70b-versatile", 
            )
            return chat.choices[0].message.content
        except Exception as e:
            return f"Groq Error: {e}"

# --- 5. Utilities (Save/Email) ---
def save_to_md(text, filename="summary.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return os.path.abspath(filename)

def send_email_func(user, password, recipient, subject, body):
    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(user, password)
        server.sendmail(user, recipient, msg.as_string())
        server.quit()
        return "Email sent!"
    except Exception as e:
        return f"Email Error: {e}"