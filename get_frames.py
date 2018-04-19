import os

import cv2

class face_detect:

    def __init__(self):
        self.frame= None

    def detect_frame(self, frame):
        gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('c', gray_img)
        face= haar_cascade.detectMultiScale(gray_img, scaleFactor= 1.1,
                                            minNeighbors= 5, minSize= (30,30), maxSize=(100,100))
        print( len(face))
        if len(face)!= 0:
            print('face_detected_bc')
            for (x, y, w, h) in face:
                try:
                    face_comp= frame[y:y+h, x:x+w, :]
                    print(face_comp.shape)
                    fc= cv2.resize(face_comp, (96,96))
                    count= 0
                    cv2.imwrite('frame_'+str(count)+'.jpg', fc)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 204, 102), 2)
                    # if cv2.waitKey(0) & 0xFF== ord('q'):
                    cv2.imshow('Frames', frame)

                    count+= 1


                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                except:
                    pass
        else:
            print('No face detected')

if __name__ == "__main__":

    fd= face_detect()

    count= 1
    path = os.getcwd()
    haar_cascade = cv2.CascadeClassifier('C:/Users/LENOVO/Anaconda3/pkgs/opencv-3.3.1-py35h20b85fd_1'
                                         '/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')
    cap= cv2.VideoCapture(path+'/../data/track.mp4')
    if cap.isOpened() == False:
        print("Video file dosen't exist")

    while cap.isOpened():

        ret, frame = cap.read()

        if ret == True:
            print('Frame: {}'.format(count))
            # cv2.imshow('Frames', frame)
            fd.detect_frame(frame)
            # exit()
            count+= 1

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    exit()