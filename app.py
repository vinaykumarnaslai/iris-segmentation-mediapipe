from flask import Flask, render_template, jsonify
import threading
import cv2 as cv
import numpy as np
import mediapipe as mp

# Create a Flask application instance single occurence
app = Flask(__name__)

# Define a route for the root URL ('/') that renders the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for starting face mesh detection. This will be triggered via a POST request.
@app.route('/start-face-mesh', methods=['POST'])
def start_face_mesh():
    # Start the face mesh detection in a separate thread to avoid blocking the main thread
    threading.Thread(target=run_face_mesh).start()
    return jsonify({"message": "Face mesh detection started"}), 200

# Function that runs the face mesh detection
def run_face_mesh():
    # Initialize MediaPipe Face Mesh solution
    mp_face_mesh = mp.solutions.face_mesh
    # Define the indices for the left and right iris landmarks
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    # Open the default camera
    cap = cv.VideoCapture(0)
    # Set the window properties to fullscreen
    cv.namedWindow('img', cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty('img', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # Use the FaceMesh class from MediaPipe with specified parameters
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Detect at most one face
        refine_landmarks=True,  # Refine landmarks for better accuracy
        min_detection_confidence=0.5,  # Minimum confidence for face detection
        min_tracking_confidence=0.5  # Minimum confidence for face tracking
    ) as face_mesh:
        while True:
            ret, frame = cap.read()  # Capture a frame from the camera
            if not ret:
                break  # Exit the loop if the frame is not captured successfully
            frame = cv.flip(frame, 1)  # Flip the frame horizontally for mirror effect
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert the frame to RGB
            img_h, img_w = frame.shape[:2]  # Get the height and width of the frame
            results = face_mesh.process(rgb_frame)  # Process the frame to detect face landmarks
            
            if results.multi_face_landmarks:
                # Extract mesh points and scale them to the frame dimensions
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                # Get the enclosing circle for the left iris
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                # Get the enclosing circle for the right iris
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)  # Center of the left iris
                center_right = np.array([r_cx, r_cy], dtype=np.int32)  # Center of the right iris
                # Draw circles around the left and right iris
                cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
            
            # Add "Press q to Quit" text with a background rectangle
            text = "Press q to Quit"
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            color = (255, 255, 255)
            thickness = 1
            bg_color = (0, 0, 0)
            text_size = cv.getTextSize(text, font, scale, thickness)[0]
            text_x = 10
            text_y = 30
            box_coords = ((text_x, text_y + 5), (text_x + text_size[0] + 10, text_y - text_size[1] - 10))
            # Draw a filled rectangle for the text background
            cv.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv.FILLED)
            # Put the text on the frame
            cv.putText(frame, text, (text_x + 5, text_y - 5), font, scale, color, thickness, cv.LINE_AA)
            
            # Display the frame in a window named 'img'
            cv.imshow('img', frame)
            key = cv.waitKey(1)  # Wait for a key press
            if key == ord('q'):  # Exit the loop if 'q' is pressed
                break

    cap.release()  # Release the camera
    cv.destroyAllWindows()  # Close all OpenCV windows

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
