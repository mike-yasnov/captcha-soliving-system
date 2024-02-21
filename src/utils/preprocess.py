import cv2

def preprocess_image(image):
    blur = cv2.GaussianBlur(image, (5,5), 0)

    # convert to hsv and get saturation channel
    sat = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)[:,:,1]

    thresh = cv2.threshold(sat, 55, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # apply morphology close and open to make mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=3)

    img_result = image.copy()
    img_result[mask!=0] = 0

    return img_result