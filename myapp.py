import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
import webbrowser
import os
import pandas as pd
import base64

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('D:/UiTM/CSP650/Project/Final/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("D:/UiTM/CSP650/Project/Final/emotion_model.h5")
print("Loaded model from disk")

def main():

    st.title("Emotion Based Music Recommendation System")
    activiteis = ["Home", "Emotion Detection","Music Recommender"]
    #Background design
    def set_bg(main_bg):
        # set bg name
        main_bg_ext = "png"
            
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
                background-size: cover

            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #FFFFFF;
            color:black;
        }
        [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
        }
        p {
        color: black;
        font-size:15px;
        }
        div.stButton > button:first-child {
        background-color: white;
        }
        h1{
        color:black;
        }
        li{
        color: black;
        }
    </style>
    """, unsafe_allow_html=True)
    
    choice = st.sidebar.selectbox("Select Page", activiteis)
    st.sidebar.markdown(
        """ Developed by Ikhmal  
            Fyp Project""")

    if choice == "Home":

        html_temp_home1 = """<div style="background-color:#0a3e81;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        set_bg("D:/UiTM/CSP650/Project/Final/bg1.jpg")
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time face detection using web cam feed.

                 2. Recommend music from emotion detected.

                 User can choose to manually select emotion or using predictive model to detect facial emotion and get the recommended music.

                 Emotion detection page will use predictive model and music recommender page will use manual selection by user.

                 """)
        
    elif choice == "Emotion Detection":
        set_bg("D:/UiTM/CSP650/Project/Final/bg1.jpg")
        st.markdown("""
        <style>
        .st-table {
        background-color: #000000;
        }
        </style>
        """, unsafe_allow_html=True)

        # Reset the "run" variable when switching to Emotion Detection
        if choice != st.session_state.get("previous_page", ""):
            st.session_state["run"] = "true"

        st.session_state["previous_page"] = choice
        # Initialize emotion variable as an empty string
        emotion = ""
        # Initialize state variables
        capture_active = False


        cap = cv2.VideoCapture(0)
        face_detector = cv2.CascadeClassifier('D:/UiTM/CSP650/Project/Final/haarcascade_frontalface_default.xml')
        text_placeholder = st.empty()
        frame_placeholder = st.empty()
        imgcapture_placeholder=st.empty()
        result_placeholder=st.empty()

        # Button to capture image
        capture_button = st.button("Capture Emotion")
        if capture_button:
            capture_active = True
            
        # Capture image if button clicked and active
        if capture_active:
            # Capture current frame
            ret, frame = cap.read()

            # Save captured frame
            cv2.imwrite("captured_image.jpg", frame)

            # Display message
            imgcapture_placeholder.write("Image captured successfully!")

            # Read captured image
            captured_image = cv2.imread("captured_image.jpg")

            # Convert to grayscale
            gray_captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

            # Face detection on captured image
            num_faces = face_detector.detectMultiScale(gray_captured_image, scaleFactor=1.3, minNeighbors=5)
            emotion=""
            # Process each detected face in the captured image
            for (x, y, w, h) in num_faces:
                # Extract ROI for face in captured image
                captured_roi_gray_frame = gray_captured_image[y:y + h, x:x + w]

                                    # Preprocess captured ROI
                captured_cropped_img = np.expand_dims(np.expand_dims(cv2.resize(captured_roi_gray_frame, (48, 48)), -1), 0)

                # Predict emotion using model
                captured_emotion_prediction = emotion_model.predict(captured_cropped_img)
                captured_maxindex = int(np.argmax(captured_emotion_prediction))
                maxindex = captured_maxindex  # Update maxindex inside the loop
                emotion=emotion_dict[captured_maxindex]

                # Calculate and display prediction accuracy
                captured_accuracy = captured_emotion_prediction[0][captured_maxindex] * 100
                
                #Changed into single parargraph
                #result_placeholder.write("Result from the image captured and predicted by emotion detection model.")
                #result_placeholder.write(f"Predicted Emotion: {emotion_dict[captured_maxindex]} with {captured_accuracy:.2f}% accuracy")
                paragraphs = [
                "Result from the image captured and predicted by emotion detection model.",
                f"Predicted Emotion: {emotion_dict[captured_maxindex]} with {captured_accuracy:.2f}% accuracy"
                ]
                result_placeholder.markdown("<br>".join(paragraphs), unsafe_allow_html=True)
                
                #Display captured face with bounding box
                cv2.rectangle(captured_image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                cv2.putText(captured_image, emotion_dict[captured_maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                st.image(captured_image, channels="BGR")
            cap.release()
            cv2.destroyAllWindows()


            mood_music = pd.read_csv("D:/UiTM/CSP650/Project/Final/data_moods.csv")
            mood_music = mood_music[['name', 'artist', 'mood']]

            # Filter the mood_music dataset based on the user's emotion
            filtered_music = None
            if emotion == "Angry" or emotion == "Disgusted" or emotion == "Fearful":
                filter1 = mood_music['mood'] == 'Calm'
                f1 = mood_music.where(filter1).dropna()
                filtered_music = f1.sample(n=10).reset_index(drop=True)

            elif emotion == "Happy" or emotion == "Neutral":
                filter1 = mood_music['mood'] == 'Happy'
                f1 = mood_music.where(filter1).dropna()
                filtered_music = f1.sample(n=10).reset_index(drop=True)

            elif emotion == "Sad":
                filter1 = mood_music['mood'] == 'Sad'
                f1 = mood_music.where(filter1).dropna()
                filtered_music = f1.sample(n=10).reset_index(drop=True)

            elif emotion == "Surprised":
                filter1 = mood_music['mood'] == 'Energetic'
                f1 = mood_music.where(filter1).dropna()
                filtered_music = f1.sample(n=10).reset_index(drop=True)

            # Display the filtered music recommendations (if any)
            if filtered_music is not None:
                table_html = filtered_music.to_html(index=False, escape=False)
                table_html = table_html.replace('<table', '<table style="border-radius: 10px; overflow: hidden; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);"')
                table_html = table_html.replace('<th>', '<th style="text-align:left; background-color:black; color:white;">')
                table_html = table_html.replace('<td>', '<td style="background-color:black; color:white;">')
                st.markdown(table_html, unsafe_allow_html=True)

            else:
                result_placeholder.write("No emotion detected.")
            restart_placeholder=st.empty()
            restart_placeholder.write("User can restart the page by clicking the restart webcam button.")    
            st.button("Restart webcam")    

        
        while True:
            # Check if the "run" variable is set to "false"
            if st.session_state["run"] == "false":
                break
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                    
                    # Flip the frame horizontally for better face detection
                    frm = cv2.flip(frame, 1)

                    # Detect faces using Haar cascade classifier
                    face_detector = cv2.CascadeClassifier('D:/UiTM/CSP650/Project/Final/haarcascade_frontalface_default.xml')
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                    # Process each detected face
                    for (x, y, w, h) in num_faces:
                        # Draw a bounding box around the face
                        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)

                        # Extract the region of interest (ROI) containing the face
                        roi_gray_frame = gray_frame[y:y + h, x:x + w]

                        # Preprocess the ROI by resizing it to 48x48 and converting it to a 4D array
                        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                    # Display the processed frame
                    text_placeholder.write("Please make sure the green box appear on your face before click the button")
                    frame_placeholder.image(frame, channels='BGR')

            else:
                break

    elif choice == "Music Recommender":
        set_bg("D:/UiTM/CSP650/Project/Final/bg1.jpg")

        st.header("Music Recommendation System")        
        mood_music = pd.read_csv("D:/UiTM/CSP650/Project/Final/data_moods.csv")
        mood_music = mood_music[['name', 'artist', 'mood']]
        # Get the user's emotion
        emotion = st.selectbox("What is your current emotion?", ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"])
        def open_youtube_search(search_term):
            webbrowser.open(f"https://www.youtube.com/results?search_query={search_term}")

        search_term = st.text_input("Enter the song title and artist:")

        if st.button("Search YouTube"):
            open_youtube_search(search_term)

        # Filter the mood_music dataset based on the user's emotion
        filtered_music = None
        if emotion == "Angry" or emotion == "Disgusted" or emotion == "Fearful":
            filter1 = mood_music['mood'] == 'Calm'
            f1 = mood_music.where(filter1).dropna()
            filtered_music = f1.sample(n=10).reset_index(drop=True)

        elif emotion == "Happy" or emotion == "Neutral":
            filter1 = mood_music['mood'] == 'Happy'
            f1 = mood_music.where(filter1).dropna()
            filtered_music = f1.sample(n=10).reset_index(drop=True)

        elif emotion == "Sad":
            filter1 = mood_music['mood'] == 'Sad'
            f1 = mood_music.where(filter1).dropna()
            filtered_music = f1.sample(n=10).reset_index(drop=True)

        elif emotion == "Surprised":
            filter1 = mood_music['mood'] == 'Energetic'
            f1 = mood_music.where(filter1).dropna()
            filtered_music = f1.sample(n=10).reset_index(drop=True)

        # Display the filtered music recommendations (if any)
        if filtered_music is not None:
            st.write(f"User emotion: {emotion}")
            table_html = filtered_music.to_html(index=False, escape=False)
            table_html = table_html.replace('<table', '<table style="border-radius: 10px; overflow: hidden; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);"')
            table_html = table_html.replace('<th>', '<th style="text-align:left; background-color:black; color:white;">')
            table_html = table_html.replace('<td>', '<td style="background-color:black; color:white;">')
            st.markdown(table_html, unsafe_allow_html=True)

        else:
            st.write("Please select a valid emotion.")

    else:
        pass


if __name__ == "__main__":
    main()