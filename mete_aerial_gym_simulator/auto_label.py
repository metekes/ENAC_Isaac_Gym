import cv2
import glob
import os

# Opencv DNN
net = cv2.dnn.readNet("yolov4_tiny_custom_best.weights", "yolov4_tiny_custom_test.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)


# Load class lists
classes = ["obstacle"]

# Read the images
frames = glob.glob("images/*.jpg")


for frame_name in frames:
    # Read the frames
    frame = cv2.imread(frame_name)

    # Object detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.7, nmsThreshold=.4)

    f = open(frame_name[:-3] + "txt","w")

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        f.write(str(class_id) + " " + str((x+w/2)/frame.shape[1]) + " " + str((y+h/2)/frame.shape[0]) + " " + str(w/frame.shape[1]) + " " + str(h/frame.shape[0]) + "\n")

    # To display the frames
    # cv2.imshow("frame", frame)
    # cv2.waitKey(10)