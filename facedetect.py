import cv2
import face_recognition
import numpy as np
import os

# directory/folder to face store
dir_path = r'./known_faces'

# Create arrays of known face encodings and their names
known_face_encodings = []

known_face_names = []

# Iterate directory & create face encodings with names
for file_path in os.listdir(dir_path):
    # check if current file_path is a file
    if os.path.isfile(os.path.join(dir_path, file_path)):
        if file_path.endswith( ('.jpeg','.png','.jpg')):
            img = face_recognition.load_image_file(dir_path + "/" +file_path)
            face_enc = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(face_enc)
            known_face_names.append(file_path.split(".")[0])

print ("%d face encoded" % len(known_face_encodings))

# define a video capture object 
vid = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
  
while(True):
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 

    # Only process every other frame of video to save time
    if process_this_frame:
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = small_frame[:, :, ::-1]

        rgb_img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_img)
        # print("Found {} faces in image.".format(len(face_locations)))
        face_encodings = face_recognition.face_encodings(rgb_img)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append("%s[%.2f]" % (name, face_distances[best_match_index]))

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if face_distances[best_match_index] > 0.55:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (100, 255, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (100, 255, 255), cv2.FILLED)
        else:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (51, 255, 51), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (51, 255, 51), cv2.FILLED)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
