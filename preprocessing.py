import cv2
import numpy as np
import imutils
from PIL import Image

def image_preparation(image_file):
    img_arr = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_file, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.threshold(blurred, 253, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    pic=cv2.bitwise_not(thresh)
    return pic, gray

def x_cord_contour(contours):
    #Returns the X cordinate for the contour centroid
    M = cv2.moments(contours)
    return (int(M['m10']/M['m00']))

def expand_image(img, weight, height):
    w, h = weight, height
    img = Image.fromarray(img)
    background = Image.new('RGB', (w + 100, h + 50), 'white')
    background.paste(img, (50, 25))
    background = np.array(background)

    return background


def finding_contours(pic, img_arr, gray):
    # find contours in the thresholded image, then initialize the digit contours lists
    cnts = cv2.findContours(pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    objs = {}
    # Sort by left to right using our y_cord_contour function
    try:
        cnts = sorted(cnts, key=x_cord_contour, reverse=False)
    except:
        return objs
    # loop over the digit area candidates
    i = 0
    x_previous, y_previous = 0, 0
    y_min = 1000
    w_mean, h_mean, y_mean  = 0, 0, 0
    for cc in cnts:
        (x, y, w, h) = cv2.boundingRect(cc)
        if y < y_min:
            y_min = y
        y_mean += y
        w_mean += w
        h_mean += h
    w_mean = w_mean / len(cnts)
    h_mean = h_mean / len(cnts)
    y_mean = y_mean / len(cnts)
    print('y', y_mean)
    for c in cnts:
        pow = False
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        if x == x_previous:
            cv2.rectangle(img_arr, (x, y), (x + w, y_previous + h), (0, 255, 0), 1)
            letter_crop = gray[y:y_previous + h, x:x + w]
            exp_img = expand_image(letter_crop, w, (h+15) * 2)
            i -= 1
        else:
            # подсчитать среднюю высоту и улучшить распознавание степени
            x_previous = x
            y_previous = y
            if (y == y_min or y <= int(y_mean)) and (w < w_mean and h < h_mean):
                cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 255, 0), 1)
                letter_crop = gray[y:y + h, x:x + w]
                exp_img = expand_image(letter_crop, w, h)
                pow = True
            else:
                cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 255, 0), 1)
                letter_crop = gray[y:y + h, x:x + w]
                exp_img = expand_image(letter_crop, w, h)
        exp_img = cv2.cvtColor(exp_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"num_{i}.png", exp_img)
        objs[i] = (x, y, w, h, pow)
        i+=1
    cv2.imwrite(f"pic.png", img_arr)
    return objs
