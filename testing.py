from tensorflow import keras
import numpy as np
import cv2
import keras
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical

dataset=np.load("Datasets/TRAINofemotions.npy",allow_pickle=True)
classes=["Angry","Happy","Normal","Sad","Surprised"]
test_inputs=[]
test_targets=[]

for image ,target in dataset[9000:]:
     test_inputs.append(image)
     test_targets.append(target)

test_inputs=np.array(test_inputs)
test_targets=np.array(test_targets)
normalised_test_inputs=test_inputs/255
test_targets=np_utils.to_categorical(test_targets, num_classes=5)
model=keras.models.load_model("Models/best_of_faceemotion.hdf5")
for i , test in enumerate(normalised_test_inputs):    # for 1 st iteration i will be 0 and test will be normalised_test_inputs[0]
     prediction=model.predict(test.reshape(1,50,50,3))
     max_index=np.argmax(prediction[0])
     emotion_prediction=classes[max_index]
     cv2.imshow("Image",test)
     print("Prediction: ",emotion_prediction)
     if cv2.waitKey(0)==27:
           break
cv2.destroyAllWindows()
model.evaluate(normalised_test_inputs,test_targets)

