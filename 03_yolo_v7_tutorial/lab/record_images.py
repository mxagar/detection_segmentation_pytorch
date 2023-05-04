import cv2

# Set camera: 0 default/laptop, 1 webcam, etc.
cap = cv2.VideoCapture(1)
# Initialize
num = 0
path = "data/captured"

while cap.isOpened():
    succes1, img = cap.read()
    k = cv2.waitKey(1)

    if k == 27: # ESC
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(path + '/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img '+str(num), img)

cap.release()
cv2.destroyAllWindows()
