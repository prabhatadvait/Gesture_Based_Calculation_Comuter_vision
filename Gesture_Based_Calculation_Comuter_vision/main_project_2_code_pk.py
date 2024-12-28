# import cv2
# import mediapipe as mp
# import numpy as np
# from tensorflow.keras.models import load_model
#
# # Load pre-trained model (replace with your trained model file)
# model = load_model('digit_symbol_model_entire_dataset.h5')
#
# # Mediapipe setup for hand tracking
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
#
# # Drawing variables
# canvas = np.zeros((480, 640, 3), np.uint8)  # Black canvas
# last_point = None  # Previous point to draw lines
# equation = ""  # Equation string to store recognized digits and symbols
#
# # Function to preprocess the drawing for digit/symbol recognition
# def preprocess_digit(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
#     bbox = cv2.boundingRect(thresh)
#     digit_roi = thresh[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
#     digit_resized = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
#     digit_resized = digit_resized / 255.0  # Normalize
#     digit_resized = np.expand_dims(digit_resized, axis=[0, -1])  # Model expects (1, 28, 28, 1)
#     return digit_resized
#
# # Function to count raised fingers
# def count_raised_fingers(hand_landmarks):
#     fingers_up = [
#         hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,   # Index finger
#         hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y, # Middle finger
#         hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y, # Ring finger
#         hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y  # Pinky finger
#     ]
#     return sum(fingers_up)
#
# # Function to segment the canvas into individual digits/symbols
# def segment_canvas(canvas):
#     gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     segments = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w > 10 and h > 10:  # Filter small noise
#             roi = canvas[y:y + h, x:x + w]
#             processed_digit = preprocess_digit(roi)
#             prediction = np.argmax(model.predict(processed_digit), axis=1)[0]
#             segments.append((x, prediction))
#
#     # Sort digits by their x-coordinate to form the correct equation
#     segments.sort(key=lambda item: item[0])
#     detected_sequence = ''.join(str(item[1]) for item in segments)
#
#     return detected_sequence
#
# # Capture video from the webcam
# cap = cv2.VideoCapture(0)
#
# with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.flip(frame, 1)  # Flip the frame horizontally
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb_frame)
#
#         # Show canvas in a separate window
#         cv2.imshow('Drawing Canvas', canvas)
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#                 # Get index finger tip coordinates (landmark 8)
#                 index_finger_tip = hand_landmarks.landmark[8]
#                 h, w, _ = frame.shape
#                 x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
#
#                 # Count raised fingers
#                 raised_fingers = count_raised_fingers(hand_landmarks)
#
#                 # If exactly one finger is raised, allow drawing
#                 if raised_fingers == 1:
#                     if last_point is not None:
#                         cv2.line(canvas, last_point, (x, y), (255, 255, 255), 5)
#                     last_point = (x, y)
#                 else:
#                     last_point = None  # Stop drawing if no valid gesture
#
#                 # Clear canvas if five fingers are raised
#                 if raised_fingers == 5:
#                     canvas = np.zeros((480, 640, 3), np.uint8)
#                     equation = ""
#
#                 # Process the canvas if four fingers are raised (indicating drawing is complete)
#                 if raised_fingers == 4:
#                     detected_equation = segment_canvas(canvas)
#                     print("Detected Equation:", detected_equation)
#                     try:
#                         result = eval(detected_equation)
#                         print("Result:", result)
#                     except Exception as e:
#                         print("Error:", e)
#                     # Clear the canvas after processing
#                     canvas = np.zeros((480, 640, 3), np.uint8)
#
#         # Display the main frame with hand landmarks
#         cv2.imshow('Gesture-based Digit Recognition', frame)
#
#         if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
#             break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model (replace with your trained model file)
model = load_model('digit_symbol_model_entire_dataset.h5')

# Mediapipe setup for hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Drawing variables
canvas = np.zeros((480, 640, 3), np.uint8)  # Black canvas
last_point = None  # Previous point to draw lines
line_thickness = 10  # Adjust line thickness to improve drawing
stored_data = []  # List to store detected digits and symbols
equation = ""  # To store the full equation
is_drawing_completed = False  # Flag to track if drawing is completed

# Function to preprocess the drawing for digit/symbol recognition
def preprocess_digit(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    bbox = cv2.boundingRect(thresh)
    digit_roi = thresh[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    digit_resized = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
    digit_resized = digit_resized / 255.0  # Normalize
    digit_resized = np.expand_dims(digit_resized, axis=[0, -1])  # Model expects (1, 28, 28, 1)
    return digit_resized

# Function to count raised fingers
def count_raised_fingers(hand_landmarks):
    fingers_up = [
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,   # Index finger
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y, # Middle finger
        hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y, # Ring finger
        hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y  # Pinky finger
    ]
    return sum(fingers_up)

# Function to draw on canvas with the hand's movements
def draw_on_canvas(last_point, current_point, canvas):
    if last_point is not None:
        cv2.line(canvas, last_point, current_point, (255, 255, 255), line_thickness)
    return current_point

# Function to map the hand's coordinates to the canvas dimensions
def map_coordinates(x, y, frame_width, frame_height, canvas_width, canvas_height):
    x_new = int(x * canvas_width / frame_width)
    y_new = int(y * canvas_height / frame_height)
    return x_new, y_new

# Function to segment the canvas into individual digits/symbols
def segment_canvas(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filter small noise
            roi = canvas[y:y + h, x:x + w]
            processed_digit = preprocess_digit(roi)
            prediction = np.argmax(model.predict(processed_digit), axis=1)[0]
            segments.append((x, prediction))

    # Sort digits by their x-coordinate to form the correct equation
    segments.sort(key=lambda item: item[0])
    detected_sequence = ''.join(str(item[1]) for item in segments)

    return detected_sequence

# Capture video from the webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Show canvas in a separate window
        cv2.imshow('Drawing Canvas', canvas)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip coordinates (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Map the coordinates to the canvas
                x_mapped, y_mapped = map_coordinates(x, y, w, h, canvas.shape[1], canvas.shape[0])

                # Count raised fingers
                raised_fingers = count_raised_fingers(hand_landmarks)

                # If exactly one finger is raised, allow drawing
                if raised_fingers == 1:
                    last_point = draw_on_canvas(last_point, (x_mapped, y_mapped), canvas)
                    is_drawing_completed = False  # Keep track of drawing state

                else:
                    last_point = None  # Stop drawing if no valid gesture

                # Clear canvas if five fingers are raised
                if raised_fingers == 5:
                    canvas = np.zeros((480, 640, 3), np.uint8)
                    equation = ""
                    stored_data = []  # Clear stored data
                    is_drawing_completed = False  # Reset drawing state

                # Process the canvas if four fingers are raised (indicating drawing is complete)
                if raised_fingers == 4 and not is_drawing_completed:
                    detected_equation = segment_canvas(canvas)
                    print("Detected Equation:", detected_equation)
                    stored_data.append(detected_equation)  # Store the recognized digit or symbol
                    canvas = np.zeros((480, 640, 3), np.uint8)  # Clear canvas after storing
                    is_drawing_completed = True  # Mark drawing as completed

                # Perform calculation if three fingers are raised
                if raised_fingers == 3 and len(stored_data) == 2:
                    try:
                        # Assuming the stored data contains two numbers and one operator
                        equation = f"{stored_data[0]}{stored_data[1]}"
                        result = eval(equation)
                        print("Result:", result)
                        stored_data = []  # Clear stored data after calculation
                        is_drawing_completed = False  # Reset drawing state
                    except Exception as e:
                        print("Error:", e)

        # Display the main frame with hand landmarks
        cv2.imshow('Gesture-based Digit Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
