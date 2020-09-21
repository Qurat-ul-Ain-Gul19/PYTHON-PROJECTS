import cv2
import numpy as np

classes = []
confidenceTreshold= 0.5
nms_threshold= 0.3

with open("files/coco.names", "rt") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
cap= cv2.VideoCapture("videos/abdullah.mp4")

wht= 416

modelConfiguration= "files/35 fps med resolution 416/yolov3.cfg"
modelWeights= "files/35 fps med resolution 416/yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights) #here we are loading the neural network
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT, wT, cT= img.shape
    bbox= []
    classIds= []
    confs= []
    for output in outputs:
        for det in output:
            scores= det[5:]
            classId= np.argmax(scores)
            confidence= scores[classId]

            if confidence > confidenceTreshold:
                w,h= int(det[2]*wT), int(det[3]*hT) #pixel values
                x,y= int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbox))
    indices= cv2.dnn.NMSBoxes(bbox, confs, confidenceTreshold, nms_threshold)

    for i in indices:
        i= i[0]
        box= bbox[i]
        x,y,w,h= box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)
        cv2.putText(img,f"{classes[classIds[i]].upper()} {int(confs[i]*100)}%",
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2)













while True:
    success, img= cap.read()

    blob= cv2.dnn.blobFromImage(img, 1/255, (wht,wht), [0,0,0], 1, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    # print(layer_names)
    # print(net.getUnconnectedOutLayers())
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(output_layers)
    outs = net.forward(output_layers)
    # print(len(outs))
    # print(type(outs))
    # print(len(outs[0]))
    # print(outs[0].shape)
    # print(outs[1].shape)
    # print(outs[2].shape)
    findObjects(outs, img)


    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break