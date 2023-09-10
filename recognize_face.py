import cv2
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_data.yml')
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

def display():
    id = 0
    names = ["None"]
    with open("name_data", "r") as name_data:   
        for i in name_data.readlines():
            names.append(i[:-1])
    print(names)
    while True:
        _, img =cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            if (confidence < 100):
                id = names[id]
                newlines = []
                with open("attendance_data.csv", "r") as att:
                    for i in att.readlines():
                        if i.split(",")[0] != id:
                            newlines.append(i)
                    newlines.append(f"{id}, {datetime.now().strftime('%d-%m-%Y, %H:%M:%S')}\n")
                with open("attendance_data.csv", "w") as att:
                    att.writelines(newlines)
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(
                        img, 
                        str(id), 
                        (x+5,y-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                    )
            cv2.putText(
                        img, 
                        str(confidence), 
                        (x+5,y+h-5), 
                        font, 
                        1, 
                        (255,255,0), 
                        1
                    )  
        
        _, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')  
        