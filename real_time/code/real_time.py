from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2

cascade_path = '../haarcascade_frontalface_default.xml'
model_path = '../checkpoints/best_model.model'

det = cv2.CascadeClassifier(cascade_path)
model = load_model(model_path)

EMOTIONS = ['Angry', 'Dosgust', 'Scared', 'Happy', 'Sad', 'Surprised', 'Neutral']

cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False
    
while ret:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 500)
    clone = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    
    rects = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), 
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(rects) > 0:
        rect = sorted(rects, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        
        (x, y, w, h) = rect
        
        face = gray[y:y+h, x:x+w]
        roi = face.copy()
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        preds = model.predict(roi)[0]
        print(preds)
        label = EMOTIONS[preds.argmax()]
        
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            l = int(prob * 300)
            cv2.rectangle(canvas, (5, (i * 35) + 5),
                          (l, (i * 35) + 35), (0, 255, 0), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
        
        cv2.putText(clone, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cv2.imshow("face", face)    
    cv2.imshow("feed" , clone)
    cv2.imshow("probabilities", canvas)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
camera.release()
cv2.destroyAllWindows()
        
        
    
    





