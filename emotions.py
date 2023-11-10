import numpy as np
import cv2
import datetime
import time
import tflite_runtime.interpreter as tflite

Interpreter=tflite.Interpreter(model_path="converted_model.tflite")
Interpreter.allocate_tensors()

input_details=Interpreter.get_input_details()
output_details=Interpreter.get_output_details()

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# start the webcam feed
cap = cv2.VideoCapture(0)
Emotions_File = open("Emotions_File.csv", "a") # Abre archivo .csv
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cropped_img=np.array(cropped_img, dtype='f')
        Interpreter.set_tensor(input_details[0]['index'], cropped_img)
        Interpreter.invoke()
        output_data = Interpreter.get_tensor(output_details[0]['index'])
        maxindex = int(np.argmax(output_data))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2, cv2.LINE_AA) # Texto de la emoción

        # ************************ Se agrega la información al archivo .csv ********************************** #
        emocion = emotion_dict[maxindex]
        tc = datetime.datetime.now()        # tc contiene el tiempo actual (hora y fecha)
        ts = time.time()                    # ts contiene el tiempo actual (segundos desde el epoch)
        Emotions_File.write(str((emocion))+";"+str(tc)+";"+str(ts)+"\n") # Guardar las emociones en un .csv con el formato "Emoción; fecha y hora; segundos".
        # **************************************************************************************************** #

    cv2.imshow('Video', cv2.resize(frame,(800,480),interpolation = cv2.INTER_CUBIC)) # Ventana de video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Emotions_File.close() # Cierra archivo de .csv
cap.release()
cv2.destroyAllWindows()
