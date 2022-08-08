from tensorflow import keras
import numpy as np
import cv2
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical


dataset=np.load("Datasets/TRAINofemotions.npy",allow_pickle=True)
classes=["Angry","Happy","Normal","Sad","Surprised"]

video=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
model=keras.models.load_model("Models/best_of_faceemotion.hdf5")
while True:
     ret,frame=video.read()
     faces=face_cascade.detectMultiScale(frame)
     for (x,y,w,h) in faces:
          roi=frame[y:y+h,x:x+w].copy()
          cv2.imwrite("webcamimages/face.jpg",roi)
          #test_image=image.load_img("Faceimages/webcamimages/face.jpg",target_size=(50,50,3))
          test_image=cv2.resize(roi,(50,50))
          test_image=np.array(test_image)
          test_image=np.expand_dims(test_image,axis=0)
          n_test_image=test_image/255
          prediction=model.predict(n_test_image)
          max_index=np.argmax(prediction[0])
          emotion_prediction=classes[max_index]
          cv2.putText(frame,emotion_prediction,((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
     cv2.imshow("result",frame)
     if cv2.waitKey(15)==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
