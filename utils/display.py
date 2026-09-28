import cv2

def show(title, image):

    cv2.imshow(title, image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()