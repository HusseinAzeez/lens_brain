"""
Somthing useless
"""
# import the necessary packages
import os
import argparse
import pickle
import face_recognition
import cv2
from lib.misc.paths import list_images

# ------------------------------------------------------------------------------
# construct the argument parser and parse the arguments
# ------------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
        help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
        help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
        help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
# ------------------------------------------------------------------------------
# grab the paths to the input images in our dataset
# ------------------------------------------------------------------------------
print("[INFO] quantifying faces...")
image_paths = list(list_images(args["dataset"]))
# ------------------------------------------------------------------------------
# initialize the list of known encodings and known names
# ------------------------------------------------------------------------------
known_encodings = []
known_names = []
# ------------------------------------------------------------------------------
# loop over the image paths
# ------------------------------------------------------------------------------
for index, image_path in enumerate(image_paths):
    # --------------------------------------------------------------------------
    # extract the person name from the image path
    # --------------------------------------------------------------------------
    print(f"[INFO] processing image {index + 1}/{len(image_paths)}")
    name = image_path.split(os.path.sep)[-2]
    # --------------------------------------------------------------------------
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    # --------------------------------------------------------------------------
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # --------------------------------------------------------------------------
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    # --------------------------------------------------------------------------
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    # --------------------------------------------------------------------------
    # compute the facial embedding for the face
    # --------------------------------------------------------------------------
    encodings = face_recognition.face_encodings(rgb, boxes)
    # --------------------------------------------------------------------------
    # loop over the encodings
    # --------------------------------------------------------------------------
    for encoding in encodings:
        # ----------------------------------------------------------------------
        # add each encoding + name to our set of known names and encodings
        # ----------------------------------------------------------------------
        known_encodings.append(encoding)
        known_names.append(name)
# ------------------------------------------------------------------------------
# dump the facial encodings + names to disk
# ------------------------------------------------------------------------------
print("[INFO] serializing encodings...")
data = {"encodings": known_encodings, "names": known_names}
with open(args["encodings"], "wb") as f:
    f.write(pickle.dumps(data))
