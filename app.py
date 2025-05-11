import streamlit as st
import cv2
import copy
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import time
import os
from collections import Counter, deque
import io
import itertools

import speech_recognition as sr #pip install SpeechRecognition
import io
import tempfile
from openai import OpenAI

import re
from dotenv import load_dotenv
import os

from moviepy import VideoFileClip, concatenate_videoclips

# Import the KeyPointClassifier from the original code
from model import KeyPointClassifier
from utils import CvFpsCalc

load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    st.title("ASL SignBridge")

    # Initialize session state for speech input
    if 'speech_text' not in st.session_state:
        st.session_state.speech_text = ""
    
    # Initialize session state for sign language videos
    if 'sign_videos' not in st.session_state:
        st.session_state.sign_videos = {}
    
    # Initialize session state for compiled video
    if 'compiled_video_path' not in st.session_state:
        st.session_state.compiled_video_path = None

    # Initialize session state for OpenAI API key\
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")
   
    # Initialize text-to-speech engine
    engine = pyttsx3.init()

    # Sidebar controls
    st.sidebar.header("Controls")
    use_brect = st.sidebar.checkbox("Show bounding rect", value=True)
    min_detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.85, 0.01)
    min_tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.8, 0.01)

    # Add gesture consistency control
    stability_threshold = st.sidebar.slider("Gesture Stability Threshold", 3, 15, 8, 1,
                                            help="Number of consistent detections required before speaking")
    stability_percentage = st.sidebar.slider("Stability Percentage", 50, 100, 80, 5,
                                             help="Percentage of consistent detections required in buffer")


    # Initialize models
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row.strip() for row in f]

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Set initial mode
    mode = 0
    number = -1

    # Store last spoken gesture to avoid repetition
    last_spoken_gesture = ""

    # Buffer to store recent gesture detections (for stability)
    gesture_buffer = []
    buffer_max_size = 15  # Keep the last 15 detections

    # Function to speak text in a separate thread
    def speak_text(text):
        def _speak():
            engine.say(text)
            engine.runAndWait()

        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()

    # Function to record speech and convert to text using OpenAI Whisper
    def record_speech():
        # Check if API key is provided
        if not st.session_state.openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar to use speech recognition.")
            return ""
            
        # Record audio...
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening... Speak now.")
            try:
                audio_data = recognizer.listen(source, timeout=10)
                st.write("Processing speech with Whisper...")
                
                # Create a more reliable temporary file path
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(temp_dir, "speech_audio.wav")
                
                # Save audio data to the WAV file
                with open(temp_file_path, "wb") as temp_audio_file:
                    temp_audio_file.write(audio_data.get_wav_data())
                
                try:
                    # Use OpenAI's Whisper API
                    client = OpenAI(api_key=st.session_state.openai_api_key)
                    with open(temp_file_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    # use regex and remove all punctuation, keep spaces
                    return re.sub(r'[^\w\s]', '', transcript.text)
                finally:
                    # Clean up - remove the temporary file
                    try:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                    except Exception as e:
                        st.write(f"Warning: Could not remove temporary file: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return ""
    
    # Function to find sign language videos for a word
    def find_sign_video(word):
        # This is a placeholder. In a real application, you would:
        # 1. Have a database of sign language videos
        # 2. Look up the video for the specific word
        # 3. Return the path to the video
        

        # For this example, we'll assume videos are in a 'sign_videos' directory
        # with filenames matching the words (e.g., "hello.mp4")
        video_dir = "sign_videos"
        video_path = os.path.join(video_dir, f"{word.lower()}.mp4")
        if os.path.exists(video_path):
            return video_path
        else:
            # If no specific video exists, you could return a default video
            # or generate one programmatically
            #st.write(f"No sign video found for '{word}'")
            return None
    
    # Function to compile all word videos into a single video
    def compile_videos(video_paths):
        try:
            if not video_paths:
                st.warning("No sign videos available to compile")
                return None
                
            # Load all video clips
            clips = []
            for path in video_paths:
                if path and os.path.exists(path):
                    try:
                        clip = VideoFileClip(path)
                        clips.append(clip)
                    except Exception as e:
                        st.warning(f"Could not load video {path}: {str(e)}")
            
            if not clips:
                st.warning("None of the videos could be loaded")
                return None
                
            # Concatenate all clips
            final_clip = concatenate_videoclips(clips)
            
            # Create output directory if it doesn't exist
            output_dir = "compiled_videos"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Generate unique filename
            output_path = os.path.join(output_dir, f"compiled_{int(time.time())}.mp4")
            
            # Write the result to a file
            final_clip.write_videofile(output_path, codec="libx264")
            
            # Close all clips to release resources
            for clip in clips:
                clip.close()
            final_clip.close()
            
            return output_path
            
        except Exception as e:
            st.error(f"Error compiling videos: {str(e)}")
            return None


    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Sign to Speech", "Speech to Sign"])
    
    with tab1:
        st.header("Sign2Speech")
        st.write("Camera")
        stframe = st.empty()
        # Display current detected gesture and stability
        gesture_display = st.empty()
        stability_display = st.empty()

        # Start button side to side with stop button
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start Camera")
        with col2:
            stop_button = st.button("Stop Camera")

    with tab2:
        st.header("Speech2Sign")
        speech_col1, speech_col2 = st.columns(2)
        
        with speech_col1:
            st.write("Speech Input")
            # Button to record speech
            if st.button("Record Speech"):
                text = record_speech()
                if text:
                    st.session_state.speech_text = text
                    st.write(f"You said: {text}")
                    
                    # Process the speech - split into words
                    words = text.split()
                    
                    # Reset the compiled video path
                    st.session_state.compiled_video_path = None
                    
                    # Collect video paths for each word
                    video_paths = []
                    for i, word in enumerate(words):
                        video_path = find_sign_video(word)
                        if video_path:
                            st.session_state.sign_videos[word] = video_path
                            video_paths.append(video_path)
                    
                    # Compile videos if we have any
                    if video_paths:
                        with st.spinner("Compiling sign language videos..."):
                            compiled_path = compile_videos(video_paths)
                            if compiled_path:
                                st.session_state.compiled_video_path = compiled_path
                                st.success("Videos compiled successfully!")
        
        with speech_col2:
            st.write("Sign Language Output")
            
            # Display the compiled video if available
            if st.session_state.compiled_video_path and os.path.exists(st.session_state.compiled_video_path):
                st.write(f"Showing compiled signs for: {st.session_state.speech_text}")
                st.video(st.session_state.compiled_video_path)
            # Otherwise show individual videos
            elif st.session_state.speech_text:
                st.write(f"Showing individual signs for: {st.session_state.speech_text}")
                
                missing_words = []
                for word in st.session_state.speech_text.split():
                    if word in st.session_state.sign_videos:
                        st.video(st.session_state.sign_videos[word])
                    else:
                        missing_words.append(word)
                
                if missing_words:
                    st.warning(f"No sign videos available for: {', '.join(missing_words)}")
           
    

    running = False

    if start_button:
        running = True

    if stop_button:
        running = False
        cap.release()
        st.write("Camera stopped")

    # Main loop
    while running:
        # FPS calculation
        fps = cvFpsCalc.get()

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        current_gesture = ""

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Get the gesture label
                if hand_sign_id < len(keypoint_classifier_labels):
                    current_gesture = keypoint_classifier_labels[hand_sign_id]

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    current_gesture,
                )

        # If hand is detected and a gesture is classified
        if current_gesture:
            # Add to buffer and maintain max size
            gesture_buffer.append(current_gesture)
            if len(gesture_buffer) > buffer_max_size:
                gesture_buffer.pop(0)

            # Count occurrences of each gesture in the buffer
            gesture_counts = Counter(gesture_buffer)

            # Get the most common gesture and its count
            most_common_gesture, most_common_count = gesture_counts.most_common(1)[0]

            # Calculate stability percentage
            stability = (most_common_count / len(gesture_buffer)) * 100

            # Display current detection and stability
            stability_text = f"Current: {current_gesture}, Most stable: {most_common_gesture} ({stability:.1f}%)"
            stability_display.text(stability_text)

            # Speak the gesture if:
            # 1. It's stable enough (appears frequently in buffer)
            # 2. It's different from last spoken gesture
            # 3. It's not "None"
            # 4. Buffer has enough samples to be reliable
            if (len(gesture_buffer) >= stability_threshold and
                    stability >= stability_percentage and
                    most_common_gesture != last_spoken_gesture and
                    most_common_gesture != "None"):
                speak_text(most_common_gesture)
                last_spoken_gesture = most_common_gesture

                # Visual feedback that we're speaking a gesture
                gesture_display.success(f"Speaking: {most_common_gesture}")
                time.sleep(0.2)  # Small delay to avoid too frequent updates
                gesture_display.empty()
        else:
            stability_display.text("No gesture detected")

        debug_image = draw_info(debug_image, fps, mode, number)

        # Display the image
        stframe.image(debug_image, channels="BGR", use_container_width=True)

        # Check for keypress
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # Handle mode changes
        number, mode = select_mode(key, mode)

    # Clean up
    cap.release()


# All supporting functions from the original code
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value > 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                 (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                 (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                 (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                 (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                 (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                 (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # Wrist 1
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # Wrist 2
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # Thumb: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # Thumb: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # Thumb: tip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # Index finger: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # Index finger: 2nd joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # Index finger: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # Index finger: tip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # Middle finger: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # Middle finger: 2nd joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # Middle finger: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # Middle finger: tip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # Ring finger: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # Ring finger: 2nd joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # Ring finger: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # Ring finger: tip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # Little finger: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # Little finger: 2nd joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # Little finger: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # Little finger: tip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                      (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                  (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def draw_info(image, fps, mode, number):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    mode_string = ['Logging Key Point']
    if mode == 1:
        cv2.putText(image, "MODE:" + mode_string[0], (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
    return image


if __name__ == "__main__":
    main()