import numpy as np
import cv2
import os

from keras import backend as K
K.set_image_data_format('channels_first')
from get_frames import *

from fr_utils import *
from inception_blocks_v2 import *

FRmodel= faceRecoModel(input_shape= (3,96,96))
print('Total Params: ', FRmodel.count_params())

database= dict()
database['210556']= list(img_to_encoding('frame_0.jpg', FRmodel))

print(database)

exit()
def verify(img_path,database, model):

    print('\n inside verify \n')

    encoding= img_to_encoding(img_path, model)

    for key in database.keys():
        print("Calculating Distance")
        dist= np.linalg.norm(encoding - database[key])

        if dist < 0.7:
            return(key)

    print("Nothing found")
    return 'unidentified'

haar_cascade = cv2.CascadeClassifier('C:/Users/LENOVO/Anaconda3/pkgs/opencv-3.3.1-py35h20b85fd_1'
                                         '/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')
path= os.getcwd()
cap = cv2.VideoCapture(path+'/../data/Video_2_cut.mp4')

# Define the codec and create VideoWriter object
frame_width= int(cap.get(3))
frame_height= int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi',fourcc, 25.0, (frame_width, frame_height))

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret==True:
      # write the flipped frame
      gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      face= haar_cascade.detectMultiScale(gray_img, minSize= (30,30), maxSize= (96,96), minNeighbors= 5)

      for (x,y,w,h) in face:
        face_comp= frame[y:y+h, x:x+w, :]
        face_comp= cv2.resize(face_comp, (96,96))

        cv2.imwrite('detect_frame.jpg', face_comp)
        identity= verify('detect_frame.jpg', database, FRmodel)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 254, 120), 2)
        cv2.putText(frame, identity, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        # cv2.imshow('frame', frame)
        out.write(frame)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          break
  else:
      break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()