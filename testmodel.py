import cv2

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if not video.isOpened():
    print("Error: Could not open camera.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Unique"]

while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    if not ret:
        print("Failed to capture frame")
        break

    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if 0 <= serial < len(name_list):
            name = name_list[serial]
        else:
            name = "Unknown"

        if conf < 50:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)  # Green frame
            cv2.putText(frame, name, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Green text
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)  # Red frame
            cv2.putText(frame, "Unknown", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red text

    cv2.imshow("UNIK RECOGNIZER", frame)

    k = cv2.waitKey(1)
    if k == ord('e'):
        break

video.release()
cv2.destroyAllWindows()

print("Dataset Collection Done.............")
