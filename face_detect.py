import cv2
import numpy as np
import sys

# 這裡的路徑要改對，不太好改，你們試了就會知道:-|
face_casecade = cv2.CascadeClassifier('C:\\Users\\Peter\\.conda\\envs\\tf2.0\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
catface_casecade = cv2.CascadeClassifier('C:\\Users\\Peter\\.conda\\envs\\tf2.0\\Library\\etc\\haarcascades\\haarcascade_frontalcatface.xml')
eye_casecade = cv2.CascadeClassifier('C:\\Users\\Peter\\.conda\\envs\\tf2.0\\Library\\etc\\haarcascades\\haarcascade_eye.xml')
# eye_casecade = cv2.CascadeClassifier('C:\\Users\\Peter\\.conda\\envs\\tf2.0\\Library\\etc\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')
# body_casecade = cv2.CascadeClassifier('C:\\Users\\Peter\\.conda\\envs\\tf2.0\\Library\\etc\\haarcascades\\haarcascade_fullbody.xml')
# print(sys.getsizeof(face_casecade))

# 底下註解掉的是辨識單張圖片的版本
'''img = cv2.imread('images.jfif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_casecade.detectMultiScale(gray, 1.3, 1)
bodys = body_casecade.detectMultiScale(gray, 1.3, 1)
for (x, y, w, h) in bodys:
    print(h)
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_casecade.detectMultiScale(roi_gray, minNeighbors=1)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# 這段是從鏡頭讀取串流畫面拿後辨識的版本
# 參考網站 https://towardsdatascience.com/how-to-detect-objects-in-real-time-using-opencv-and-python-c1ba0c2c69c0
imcap = cv2.VideoCapture(0)
imcap.set(3, 1280)
imcap.set(4, 720)

while True:
    if not imcap.isOpened():
        imcap.open(-1)

    success, img = imcap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not success:
        print("完全沒有畫面。完全沒有畫面。完全沒有畫面。")
        break

    faces = catface_casecade.detectMultiScale(gray, 1.3, 8)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_casecade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

    cv2.imshow('face_detect', img)
    # loop will be broken when 'q' is pressed on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
imcap.release()
cv2.destroyWindow('face_detect')

