import os
import hashlib
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import speech_recognition as sr
from gtts import gTTS
import cv2
import numpy as np
import librosa
from moviepy.editor import ImageSequenceClip, AudioFileClip
import tempfile
import threading
import queue
import subprocess
import time
import re
import platform
import signal

# Load environment variables
load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Global state management
class ChatState:
    def __init__(self):
        self.is_listening = False
        self.processing_active = False
        self.history = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.current_process = None
        self.current_user = None
        
    def update_history(self, user_input, ai_response):
        with self.lock:
            self.history.append((user_input, ai_response))
            
    def get_history(self):
        with self.lock:
            return self.history.copy()
        
    def stop_video(self):
        with self.lock:
            if self.current_process and self.current_process.poll() is None:
                if platform.system() == "Windows":
                    os.kill(self.current_process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=1)
                except:
                    pass
                self.current_process = None

chat_state = ChatState()
response_queue = queue.Queue()
audio_lock = threading.Lock()

# AI and Media functions
model = genai.GenerativeModel('gemini-2.0-flash')

def sanitize_text(text):
    """Remove special characters and markdown from text"""
    text = re.sub(r'\*+', '', text)  # Remove asterisks
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
    text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'\b(STOP|Stop|stop)\b', '', text)  # Remove stop commands
    return text.strip()

def play_text_to_speech(text):
    try:
        with audio_lock:
            natural_text = text.replace('.', '. ').replace(',', ', ')
            tts = gTTS(text=natural_text, lang="en", slow=False, lang_check=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tts.save(f.name)
                return f.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def handle_gemini_response(user_input):
    if user_input.lower().strip() in ["stop", "halt", "cancel", "exit"]:
        return "[SYSTEM] Stopping current operation..."

    try:
        history = chat_state.get_history()
        formatted_history = []
        for user_msg, ai_msg in history:
            if user_msg:
                formatted_history.append({"parts": [{"text": user_msg}], "role": "user"})
            if ai_msg and not ai_msg.startswith("[SYSTEM]"):
                formatted_history.append({"parts": [{"text": ai_msg}], "role": "model"})
        
        chat = model.start_chat(history=formatted_history)
        response = chat.send_message(user_input)
        return sanitize_text(response.text)
    except Exception as e:
        print(f"AI Error: {e}")
        return "Sorry, I'm having trouble responding."

def generate_video_from_audio(text):
    if not text or text.startswith("[SYSTEM]"):
        return None
    
    audio_path = play_text_to_speech(text)
    if not audio_path:
        return None
    
    try:
        y, sr = librosa.load(audio_path)
        frame_length = int(sr / 30)
        hop_length = frame_length // 2
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        if len(rms) == 0 or (np.max(rms) - np.min(rms)) < 1e-10:
            rms_normalized = np.zeros_like(rms)
        else:
            rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-10)
        
        num_video_frames = int(len(y) / sr * 30)
        if num_video_frames <= 0:
            return None
            
        rms_interp = np.interp(np.linspace(0, len(rms), num_video_frames), np.arange(len(rms)), rms_normalized)
        
        image1 = cv2.imread('indiagirl-split (1).jpg', cv2.IMREAD_UNCHANGED)
        image2 = cv2.imread('indiagirl-split (2).jpg', cv2.IMREAD_UNCHANGED)
        image3 = cv2.imread('indiagirl-split (1).jpg', cv2.IMREAD_UNCHANGED)
        
        target_shape = image1.shape[:2]
        image2 = cv2.resize(image2, (target_shape[1], target_shape[0]))
        image3 = cv2.resize(image3, (target_shape[1], target_shape[0]))
        
        blended_imgs = []
        for amp in rms_interp:
            if amp < 0.5:
                alpha = amp * 2
                blended_img = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
            else:
                alpha = (amp - 0.5) * 2
                blended_img = cv2.addWeighted(image2, 1 - alpha, image3, alpha, 0)
            
            blended_img_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
            blended_imgs.append(blended_img_rgb)
            
        clip = ImageSequenceClip(blended_imgs, fps=30)
        audio_clip = AudioFileClip(audio_path)
        clip = clip.set_audio(audio_clip)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            output_video_path = temp_video_file.name
            clip.write_videofile(output_video_path, codec='libx264', fps=30, audio_codec='aac')
        
        return output_video_path
    except Exception as e:
        print(f"Video Generation Error: {e}")
        return None

def play_video_with_audio(video_path):
    try:
        chat_state.stop_video()  # Stop any currently playing video
        
        if platform.system() == "Windows":
            chat_state.current_process = subprocess.Popen(
                ['ffplay', '-window_title', 'AI Assistant', '-autoexit', video_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            chat_state.current_process = subprocess.Popen(
                ['ffplay', '-window_title', 'AI Assistant', '-autoexit', video_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    except Exception as e:
        print(f"Error playing video: {e}")

def process_responses():
    while chat_state.processing_active:
        try:
            if chat_state.stop_event.is_set():
                response_queue.queue.clear()
                chat_state.stop_video()
                chat_state.stop_event.clear()
                continue

            response = response_queue.get(timeout=1)
            if response.startswith("[SYSTEM]"):
                print(f"System command: {response}")
                continue

            video_path = generate_video_from_audio(response)
            if video_path and not chat_state.stop_event.is_set():
                play_video_with_audio(video_path)
            response_queue.task_done()
        except queue.Empty:
            continue

def continuous_listening_thread(chat_queue):
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    while chat_state.is_listening:
        with sr.Microphone() as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=8)
                
                try:
                    user_input = recognizer.recognize_google(audio)
                    if any(cmd in user_input.lower() for cmd in ["stop", "halt", "cancel"]):
                        chat_queue.put(("system", "stop"))
                    else:
                        chat_queue.put(("user_input", user_input))
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Listening error: {e}")
                time.sleep(0.5)

def handle_stop_command():
    chat_state.is_listening = False
    chat_state.stop_event.set()
    response_queue.queue.clear()
    chat_state.stop_video()  # Immediately stop the video
    chat_state.update_history(None, "🛑 Operation stopped. Ready for new commands.")
    return chat_state.get_history()

def start_stop_listening(start_button_text, history):
    chat_state.history = history.copy() if history else []
    
    chat_state.is_listening = not chat_state.is_listening
    new_button_text = "Stop Voice Chat" if chat_state.is_listening else "Start Voice Chat"
    
    if chat_state.is_listening:
        if not chat_state.processing_active:
            chat_state.processing_active = True
            processing_thread = threading.Thread(target=process_responses)
            processing_thread.daemon = True
            processing_thread.start()
        
        chat_queue = queue.Queue()
        listening_thread = threading.Thread(target=continuous_listening_thread, args=(chat_queue,))
        listening_thread.daemon = True
        listening_thread.start()
        
        chat_state.update_history(None, "🎤 Listening... Speak now.")
        
        def process_chat_queue():
            while chat_state.is_listening:
                try:
                    if not chat_queue.empty():
                        msg_type, msg_content = chat_queue.get()
                        if msg_type == "system" and msg_content == "stop":
                            return handle_stop_command()
                        elif msg_type == "user_input":
                            ai_response = handle_gemini_response(msg_content)
                            chat_state.update_history(msg_content, ai_response)
                            if not ai_response.startswith("[SYSTEM]"):
                                response_queue.put(ai_response)
                except Exception as e:
                    print(f"Processing error: {e}")
                time.sleep(0.1)
            return chat_state.get_history()
        
        history = process_chat_queue()
    else:
        history = handle_stop_command()
    
    return new_button_text, history

def main_app():
    with gr.Blocks(title="AI Assistant", theme="soft") as app:
        # Authentication Section
        with gr.Column(visible=True) as auth_col:
            with gr.Tabs() as auth_tabs:
                with gr.TabItem("Register"):
                    gr.Markdown("Click the register button to continue")
                    reg_btn = gr.Button("Register", variant="primary")

                with gr.TabItem("Login"):
                    gr.Markdown("Click the login button to continue")
                    login_btn = gr.Button("Login", variant="primary")

        # Chat Interface
        with gr.Column(visible=False) as chat_col:
            chatbot = gr.Chatbot(height=500, bubble_full_width=False)
            with gr.Row():
                text_input = gr.Textbox(label="Type a message", placeholder="Type your message here...", container=False)
                send_btn = gr.Button("Send", variant="primary")
            
            with gr.Row():
                voice_btn = gr.Button("Start Voice Chat")
                clear_btn = gr.ClearButton([chatbot, text_input], variant="secondary")
            
            status_indicator = gr.Textbox(label="Status", value="Ready", interactive=False)

        # Event handlers
        def handle_text_input(text, history):
            if not text.strip():
                return "", history
                
            ai_response = handle_gemini_response(text)
            chat_state.update_history(text, ai_response)
            
            if text.lower().strip() == "stop":
                history = handle_stop_command()
            elif not ai_response.startswith("[SYSTEM]"):
                video_path = generate_video_from_audio(ai_response)
                if video_path:
                    play_video_with_audio(video_path)
            
            return "", chat_state.get_history()
        
        text_input.submit(
            handle_text_input,
            [text_input, chatbot],
            [text_input, chatbot]
        )
        
        send_btn.click(
            handle_text_input,
            [text_input, chatbot],
            [text_input, chatbot]
        )

        voice_btn.click(
            start_stop_listening,
            [voice_btn, chatbot],
            [voice_btn, chatbot]
        )

        # Dummy registration/login logic that just shows the chat interface
        def show_chat():
            return gr.update(visible=False), gr.update(visible=True)
        
        reg_btn.click(show_chat, None, [auth_col, chat_col])
        login_btn.click(show_chat, None, [auth_col, chat_col])

    return app

if __name__ == "__main__":
    # Verify ffplay is available
    try:
        subprocess.run(["ffplay", "-version"], check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print("Error: ffplay not found. Please install FFmpeg.")
        exit(1)
        
    app = main_app()
    app.launch()
