import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

maskNet = load_model("model")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    workingframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    workingframe = cv2.resize(src=workingframe, dsize=(224, 224))
    workingframe = workingframe.reshape(1, 224, 224, 3)
    #workingframe = img_to_array(workingframe)
    workingframe = preprocess_input(workingframe)
    workingframe = np.array(workingframe)
    prediction = maskNet.predict(workingframe, batch_size=1)
    print(prediction)
    (mask, withoutmask) = prediction[0]
    if mask > withoutmask:
        label = "Mask"
        color = (0, 255, 0)
    else:
        label = "WithOutMask"
        color = (0, 0, 255)
    print(label)
    label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
    #coord = (frame.shape[0]-len(label), frame.shape[1]+len(label))
    coord = (20,50)
    cv2.putText(frame, label, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Result", frame)

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
