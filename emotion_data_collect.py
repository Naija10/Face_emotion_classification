import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier(r"C:\Users\Naija\Desktop\computer vision part 2\Day 6\Codes\Cascades\haarcascade_frontalface_default.xml")
video=cv2.VideoCapture(0)
pause=True
count=0

while True:
     ret,frame=video.read()
     faces=face_cascade.detectMultiScale(frame)
     for (x,y,w,h) in faces:
          roi=frame[y:y+h,x:x+w].copy()
          if not pause:
               count+=1
               cv2.imwrite("Faceimages/Surprised/"+str(count)+".jpg",roi)
          
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          break
     #cv2.imshow("ROI",roi)
     cv2.imshow("Frame",frame)
     key=cv2.waitKey(15)
     if key==27:
          break
     if key==ord('p'):
          pause=not(pause)
     print(count)
     if(count==2100):
          break
     
cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
