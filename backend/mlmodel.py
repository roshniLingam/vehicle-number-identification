from flask import Flask, request, jsonify
from PIL import Image
from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

class Model:
    def __init__(self):
        self.minAR=4
        self.maxAR=5
        self.debug=False

    def debug_imshow(self, title, image, waitKey=True):
        # check to see if we are in debug mode, and if so, show the
        # image with the supplied title
        if self.debug:
            cv2.imshow(title, image)

            if waitKey:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        # perform a blackhat morphological operation that will allow
        # us to reveal dark regions (i.e., text) on light backgrounds
        # (i.e., the license plate itself)
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_imshow("Blackhat", blackhat)
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions", light)
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
                        dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr", gradX)
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_imshow("Grad Erode/Dilate", thresh)
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        # return the list of contours
        return cnts

    def locate_license_plate(self, gray, candidates, clearBorder=True):
        # initialize the license plate contour and ROI
        lpCnt = None
        roi = None
        # loop over the license plate candidate contours
        for c in candidates:
            # compute the bounding box of the contour and then use
            # the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if ar >= self.minAR and ar <= self.maxAR:
                # store the license plate contour and extract the
                # license plate from the grayscale image and then
                # threshold it
                lpCnt = c
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                
                # check to see if we should clear any foreground
                # pixels touching the border of the image
                # (which typically, not but always, indicates noise)
                if clearBorder:
                    roi = clear_border(roi)
                # display any debugging information and then break
                # from the loop early since we have found the license
                # plate region
                self.debug_imshow("License Plate", licensePlate)
                self.debug_imshow("ROI", roi)
                break
        # return a 2-tuple of the license plate ROI and the contour
        # associated with it
        return (roi, lpCnt)

    def build_tesseract_options(self, psm=7):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        # return the built options string
        return options

    def find_and_ocr(self, image, psm=7, clearBorder=False):
        # initialize the license plate text
        lpText = None
        # convert the input image to grayscale, locate all candidate
        # license plate regions in the image, and then process the
        # candidates, leaving us with the *actual* license plate
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        (lp, lpCnt) = self.locate_license_plate(gray, candidates, 
                                                clearBorder=clearBorder)
        # only OCR the license plate if the license plate ROI is not
        # empty
        
        if lp is not None:
            # OCR the license plate
            self.debug_imshow('pretessaract', lp)
            pytesseract.pytesseract.tesseract_cmd = r'C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
            options = self.build_tesseract_options(psm=psm)
            lpText = pytesseract.image_to_string(lp, config=options)
            self.debug_imshow("License Plate", lp)
        
        # return a 2-tuple of the OCR'd license plate text along with
        # the contour associated with the license plate region
        return (lpText, lpCnt)

    def cleanup_text(self, text):
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV
        return "".join([c if ord(c) < 128 else "" for c in text]).strip()

    def process_image(self):
        if 'image' not in request.files:
            print('No file found')
            return jsonify('No file found')
        
        file = request.files['image']
        print("file = ", file)
        # Read the image via file.stream
        img = Image.open(file.stream).convert('RGB')
        
        image = np.array(img)
        image = cv2.resize(image, (600, 360))

        (lpText, lpCnt) = self.find_and_ocr(image)
        

        if lpText is not None and lpCnt is not None:
            # fit a rotated bounding box to the license plate contour and
            # draw the bounding box on the license plate
            box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
            box = box.astype("int")
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
            
            # compute a normal (unrotated) bounding box for the license
            # plate and then draw the OCR'd license plate text on the
            # image
            (x, y, w, h) = cv2.boundingRect(lpCnt)
            cv2.putText(image, self.cleanup_text(lpText), (x, y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # show the output ANPR image
            print("[INFO] {}".format(lpText))
            cv2.imwrite("images/output.png", image)
            return jsonify({'vehicleId':format(self.cleanup_text(lpText))})
        else:
            return jsonify({'vehicleId':'unable to decode'})
