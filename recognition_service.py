import numpy as np
import pickle
import cv2
from keras.models import load_model


def for_exponentiation(formula):
    help = ""
    while (formula.find('^') != -1):
        help += formula[0:formula.find('^') + 1]
        formula = formula[formula.find('^') + 1:len(formula)]
        help0 = ""
        try:
            while ((formula[0] != "+") and (formula[0] != "-") and
                (formula[0] != "/") and (formula[0] != "*") and (formula[0] != "=")):
                help0 += formula[0]
                formula = formula[1:len(formula)]
            if len(help0) == 1:
                help += help0
            else:
                help += "{" + help0 + "}"
        except:
            if len(help0) == 1:
                help += help0
            else:
                help += "{" + help0 + "}"
    help += formula
    return help


def for_sqrt(formula):
    help = ""
    while (formula.find('sqrt') != -1):
        help += formula[0:formula.find('sqrt') + 4]
        formula = formula[formula.find('sqrt') + 4:len(formula)]
        help0 = ""
        try:
            while ((formula[0] != "+") and (formula[0] != "-") and
                (formula[0] != "/") and (formula[0] != "*") and (formula[0] != "=")):
                help0 += formula[0]
                formula = formula[1:len(formula)]
            if len(help0) == 1:
                help += help0
            else:
                help += "(" + help0 + ")"
        except:
            if len(help0) == 1:
                help += help0
            else:
                help += "(" + help0 + ")"
    help += formula
    return help


def recognize_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (32, 32))
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)

    model = load_model("model/CNN/")
    lb = pickle.loads(open("model/CNN/label_bin", "rb").read())
    preds = model.predict(image)
    i = preds.argmax(axis=1)[0]
    label = lb.classes_
    text = "{}".format(label[i])
    return text

