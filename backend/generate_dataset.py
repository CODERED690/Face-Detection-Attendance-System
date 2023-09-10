import cv2
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate(name):
    with open("name_data", "r") as names:
        face_id = len(names.readlines())+1
    with open("name_data", "a") as names:
        names.write(name+'\n')
        
    print(face_id)
    print("Capturing face...")
    count = 0
    while True:
        _, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                        str(count) + ".jpg", gray[y:y+h,x:x+w])
            _, buffer = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        if count >= 200:
            break

    img = np.zeros((350, 500, 3), dtype = np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return print("Face captured!")