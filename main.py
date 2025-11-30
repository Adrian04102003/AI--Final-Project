import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import time
import warnings
import queue

# Speech Queue Setup
speech_queue = queue.Queue()
# Initialize global engine to None
global_engine = None 

def speech_handler():
    global global_engine
    try:
        global_engine = pyttsx3.init()
        print("TTS Engine Initialized in background thread.")
        
        voices = global_engine.getProperty('voices')
        if voices:
             global_engine.setProperty('voice', voices[0].id) 
    except Exception as e:
        print(f"FATAL TTS INIT ERROR: {e}. Speech disabled.")
        return

    while True:
        text = speech_queue.get() # Waits here until a command arrives
        if text is None: # Exit command
            break
        
        print(f"Speaking from queue: {text}")
        try:
            global_engine.say(text)
            global_engine.runAndWait()
        except Exception as e:
            print(f"TTS Runtime Error: {e}")
            
    if global_engine:
        global_engine.stop()

# Speech Function
def speak_text(text):
    if text:
        speech_queue.put(text)
    
threading.Thread(target=speech_handler, daemon=True).start()

# Cleanup function
def on_closing():
    print("Closing application...")
    # Signal the speech handler thread to stop
    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: 'model.p' not found. Please ensure the model file is in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Warning: Could not initialize TTS engine. Speak functionality may not work. Error: {e}")
    engine = None

voices = engine.getProperty('voices')
if voices:
    engine.setProperty('voice', voices[0].id) 
    print(f"Using voice: {voices[0].name}")

# Labels Dictionary and Expected Features
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: ' ', 37: '.'
}
expected_features = 42

stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""
last_registered_time = time.time()

# Registration delay in seconds
registration_delay = 3.0



# GUI
root = tk.Tk()
root.title("Sign Language to Speech Conversion")
root.geometry("700x680")
root.resizable(False, False)

main_bg = "#201b2d"
label_fg = "#E6E6FA"  # Lavender
title_fg = "#9400D3"  # Dark Violet
accent_fg_1 = "#A020F0" # Purple
accent_fg_2 = "#9370DB" # Medium Purple
accent_fg_3 = "#D8BFD8" # Thistle
border_color = "#4B0082" # Indigo
button_bg = "#5D3FD3"  # Iris
button_fg = "#E6E6FA"  # Lavender
button_active_bg = "#4B0082" # Indigo

root.configure(bg=main_bg)

current_alphabet = StringVar(value="--")
current_word = StringVar(value="--")
current_sentence = StringVar(value="--")
is_paused = StringVar(value="False")

title_label = Label(root, text="Sign Language to Speech Conversion", font=("Arial", 24, "bold"), fg=title_fg, bg=main_bg)
title_label.grid(row=0, column=0, pady=(5, 5))

video_border_frame = Frame(root, bg=border_color, bd=0)
video_border_frame.grid(row=1, column=0, padx=10, pady=(0, 5)) # Reduced padding

video_frame = Frame(video_border_frame, bg="black", width=640, height=360)
video_frame.pack(padx=2, pady=2)
video_frame.pack_propagate(False)

video_label = Label(video_frame, bg="black")
video_label.pack(expand=True)

content_frame = Frame(root, bg=main_bg)
content_frame.grid(row=2, column=0, padx=10, pady=(10, 5), sticky="ew")
content_frame.grid_columnconfigure((0, 1, 2), weight=1)

alpha_frame = Frame(content_frame, bg=main_bg)
alpha_frame.grid(row=0, column=0)
Label(alpha_frame, text="Current Alphabet", font=("Arial", 18), fg=label_fg, bg=main_bg).pack(pady=(0, 5))
Label(alpha_frame, textvariable=current_alphabet, font=("Arial", 22, "bold"), fg=accent_fg_1, bg=main_bg).pack()

word_frame = Frame(content_frame, bg=main_bg)
word_frame.grid(row=0, column=1)
Label(word_frame, text="Current Word", font=("Arial", 18), fg=label_fg, bg=main_bg).pack(pady=(0, 5))
Label(word_frame, textvariable=current_word, font=("Arial", 18), fg=accent_fg_2, bg=main_bg, wraplength=250).pack()

sentence_frame = Frame(content_frame, bg=main_bg)
sentence_frame.grid(row=0, column=2)
Label(sentence_frame, text="Current Sentence", font=("Arial", 18), fg=label_fg, bg=main_bg).pack(pady=(0, 5))
Label(sentence_frame, textvariable=current_sentence, font=("Arial", 18), fg=accent_fg_3, bg=main_bg, wraplength=250).pack()


button_frame = Frame(root, bg=main_bg)
button_frame.grid(row=3, column=0, pady=(10, 10))
button_frame.grid_columnconfigure((0, 1, 2), weight=1)

button_style = {
    "font": ("Arial", 14), # Reduced font
    "bg": button_bg,
    "fg": button_fg,
    "activebackground": button_active_bg,
    "activeforeground": button_fg,
    "relief": "flat",
    "height": 2,
    "width": 14, # Reduced width
    "bd": 0,
    "highlightthickness": 0
}

def reset_sentence():
    global word_buffer, sentence, stabilization_buffer
    word_buffer = ""
    sentence = ""
    stabilization_buffer = []
    current_word.set("--")
    current_sentence.set("--")
    current_alphabet.set("--")

def toggle_pause():
    if is_paused.get() == "False":
        is_paused.set("True")
        pause_button.config(text="Play")
    else:
        is_paused.set("False")
        pause_button.config(text="Pause")

Button(button_frame, text="Reset Sentence", command=reset_sentence, **button_style).grid(row=0, column=0, padx=10)
pause_button = Button(button_frame, text="Pause", command=toggle_pause, **button_style)
pause_button.grid(row=0, column=1, padx=10)
speak_button = Button(button_frame, text="Speak Sentence", command=lambda: speak_text(current_sentence.get()), **button_style)
speak_button.grid(row=0, column=2, padx=10)



# Video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    root.destroy()
    exit()

# Set camera feed to a matching resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Use 4:3 input, we will resize to 16:9

def process_frame():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time

    if is_paused.get() == "True":
        root.after(10, process_frame)
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        root.destroy()
        return

    frame = cv2.flip(frame, 1) # Flip horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    display_alphabet = current_alphabet.get()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) < expected_features:
                data_aux.extend([0] * (expected_features - len(data_aux)))
            elif len(data_aux) > expected_features:
                data_aux = data_aux[:expected_features]

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_character = None

            if predicted_character:
                display_alphabet = predicted_character
                
                stabilization_buffer.append(predicted_character)
                if len(stabilization_buffer) > 20: # Smaller buffer (approx 0.66s)
                    stabilization_buffer.pop(0)

                if stabilization_buffer.count(predicted_character) > 15: # Lower threshold
                    current_time = time.time()
                    
                    if predicted_character != stable_char or (current_time - last_registered_time > registration_delay):
                        stable_char = predicted_character
                        last_registered_time = current_time
                        current_alphabet.set(stable_char)

                        if stable_char == ' ':
                            if word_buffer.strip():
                                speak_text(word_buffer)
                                sentence += word_buffer + " "
                                current_sentence.set(sentence.strip())
                            word_buffer = ""
                            current_word.set("--")
                        elif stable_char == '.':
                            if word_buffer.strip():
                                speak_text(word_buffer)
                                sentence += word_buffer + "."
                                current_sentence.set(sentence.strip())
                            word_buffer = ""
                            current_word.set("--")
                        else:
                            word_buffer += stable_char
                            current_word.set(word_buffer)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

    # Draw predicted alphabet on the video feed
    cv2.putText(frame, f"Prediction: {display_alphabet}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    frame_resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)

# Start Application
print("Starting video processing...")
process_frame()

def on_closing():
    print("Closing application...")
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()