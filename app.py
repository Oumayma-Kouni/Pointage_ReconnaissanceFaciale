from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime

app = Flask(__name__)
camera = cv2.VideoCapture(0)
# Load a sample picture and learn how to recognize it.
Zied_image = face_recognition.load_image_file("Images/Zied-Bachoual.jpg")
Zied_face_encoding = face_recognition.face_encodings(Zied_image)[0]
Oumayma_image = face_recognition.load_image_file("Images/Oumayma-Kouni.jpg")
Oumayma_face_encoding = face_recognition.face_encodings(Oumayma_image)[0]

#Zied_image = face_recognition.load_image_file("Images/Zied-Bachoual'.jpg.")
#Zied_face_encoding = face_recognition.face_encodings(Zied_image)[1]
#Zied_image = face_recognition.load_image_file("Images/Zied_Bachoual.jpg")
#Zied_face_encoding = face_recognition.face_encodings(Zied_image)[2]
# Create arrays of known face encodings and their names
known_face_encodings = [
    Zied_face_encoding,
    Oumayma_face_encoding,

]
known_face_names = [
    "Zied Bachoual",
    "Oumayma Kouni",

]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def Entrer(name):
    with open('Entree.csv', 'r+') as f :
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            # nameList contains list of name which are before , entry[0]
            nameList.append(entry[0])
        # check if the current name in exist or not
        if name not in nameList:
            # si la personne not exist dans le fichier alors on va l'ajouter avec l'heure d'arrivee
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def Sortie(name):
    with open('Sortie.csv', 'r+') as f :
        # if anyone is already arrived man3awedch nsajlou
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            # data separee par des ,
            entry = line.split(',')
            # nameList contains list of name which are before , entry[0]
            nameList.append(entry[0])
        # check if the current name in exist or not
        if name not in nameList:
            # si la personne not exist dans le fichier alors on va l'ajouter avec l'heure d'arrivee
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


def gen_frames_Entrer():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (58, 137, 35), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (58, 137, 35), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
                Entrer(name)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_Sortir():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Only process every other frame of video to save time
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (58, 137, 35), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (58, 137, 35), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
                Sortie(name)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')
@app.route('/Entree.html')
def Entree():
    return render_template('Entree.html')
@app.route('/Sortir.html')
def Sortir():
    return render_template('Sortir.html')

@app.route('/video_feed_Entrer')
def video_feed_Entrer():
    return Response(gen_frames_Entrer(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_Sortir')
def video_feed_Sortir():
    return Response(gen_frames_Sortir(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/index.html')
def indexEntree():
    return render_template('index.html')
@app.route('/index.html')
def indexSortir():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
