# USAGE
# python yolo.py  --yolo yolo-coco
# import the necessary packages
import numpy as np
import argparse
import time
import glob
import cv2
import os
l=[]
for i in range(0, 64):
    l.append(i)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
LABELS1 = ["lion", "building", "monument", "dinosaur", "flower", "mountain", "food"]
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
COLORS1 = np.random.randint(0, 255, size=(len(LABELS1), 3), dtype="uint8")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
weightsPath1 = os.path.sep.join([args["yolo"], "yolov31.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
configPath1 = os.path.sep.join([args["yolo"], "yolov31.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading  from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net1 = cv2.dnn.readNetFromDarknet(configPath1, weightsPath1)
# load our input image and grab its spatial dimensions
cv_img = []

#for img in glob.glob("C:\\Users\\tanka\\yolo-object-detection\\image.orig\\*.jpg"):
for img in glob.glob("C:\\Users\\tanka\\yolo-object-detection\\images\\*.jpg"):
    # for img in glob.glob("E:\\images\\*.jpg"):
    image = cv2.imread(img)
    cv_img.append(image)
    (H, W) = image.shape[:2]
    # determine only the output layer names that we need from YOLO
    ln = net.getLayerNames()
    ln1 = net1.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    ln1 = [ln1[j[0] - 1] for j in net1.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    blob1 = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    net1.setInput(blob1)
    start = time.time()
    layerOutputs = net.forward(ln)
    layerOutputs1 = net1.forward(ln1)
    end = time.time()
    # show timing information on YOLO
    print("[INFO] YOLO  {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    histog = []
    histob= []
    histof=[]

    # loop over each of the layer outputs

    for output in layerOutputs1:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.4:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    lab = []
    conf = []
    bb = []
    e = {}
    dict1 = {}
    main = {}
    f = []
    f1 = []
    f2 = []
    histor = []
    box=[]
    lbox=[]
    boxdic={}
    print(idxs,"v,dfvfdv")

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            lbox=[x,y,w,h]
            box.append(list(lbox))
            color = [int(c) for c in COLORS1[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS1[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # print((crop_img))
            bb1 = ((w * h) / (image.shape[0] * image.shape[1]))
            bb.append(bb1)
            lab.append(LABELS1[classIDs[i]])
            conf.append(confidences[i])
            area = dict(zip(bb, lab))
            dic = dict(zip(conf, lab))
            dic1 = dict(zip(bb, lab))
        for k1 in range(0, len(idxs)):
            boxdic[k1] = [lab[k1], box[k1]]
        for k in range(0, len(lab)):
                x = abs(box[k][0])
                y = abs(box[k][1])
                w = abs(box[k][2])
                h = abs(box[k][3])
                #print(x, y, w, h)
                crop_img = image[y:y + h, x:x + w]
                src = crop_img
                #histr = cv2.calcHist([src], [0], None, [256], [0, 256])
                #histg = cv2.calcHist([src], [1], None, [256], [0, 256])
                #histb = cv2.calcHist([src], [2], None, [256], [0, 256])
                r= src[:, :, 1]
                #print(r)
                out = np.divide(r, 4)
                z =np.round(out)
                rz=z.astype(np.int64)
                np.histogram(rz.flatten(), bins=l)
                hist, bins = np.histogram(rz.flatten(), bins=l)
                print(hist,"fbbgfdf--")
                r1 = src[:, :, 1]
                out1 = np.divide(r1, 4)
                z1 = np.round(out1)
                zr1 = z1.astype(np.int64)
                np.histogram(zr1.flatten(), bins=l)
                hist1, bins = np.histogram(zr1.flatten(), bins=l)
                # histog.append(list(hist1))
                r2 = src[:, :, 2]
                out2 = np.divide(r2, 4)
                z2 = np.round(out2)
                zr2 = z2.astype(np.int64).flatten()
                np.histogram(zr2.flatten(), bins=l)
                hist2, bins = np.histogram(zr2.flatten(), bins=l)
                # histob.append(list(hist2))python yolo.py  --yolo yolo-coco

                dict1[k] = [lab[k], list(hist), list(hist1), list(hist2)]
        #print(dict1)

                #if (len(dict1)==len(lab)):
                #l = []

                #print(histor)
                #print(histog)
                #print(histob)
            #print(boxdic)
            #for i in boxdic.keys():
                #boxdic[i]

        dictionary = dict(zip(bb, zip(lab, conf)))
        print("MODIFED WITH 50% PROBALITY ,WILL SHOW CLASS NAME W.R.T COUNT--custom")
        print(str(img))
        for j in range(0, len(lab)):
            count = 0
            a = 0
            for m in area:
                if lab[j] == area[m]:
                    count += 1
                    a = a + m
                    if (a >= 1.0):
                        a = 1
            main[lab[j]] = [round(a,1), count]
        print(main)


        # from scipy.spatial import distance
        # distance.euclidean([1, 0, 0], [0, 1, 0])
        #s = 0
        #print(main,"------------------------")
        #print(dict1,"------------------------")
        fname = str(img)
        V = fname
        V = str(main) +"#"+str(dict1)+"#"+fname
        f = open("features.txt", "a")
        f.write(V+" \n")
        f.close()
        f = open("features.txt", "r")
        for r in f:
            l1 = list(r.split('*'))
                # print(l)
        f.close()
    #if (len(idxs))==len(box):

    else:
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        lab = []
        conf = []
        bb = []
        e = {}
        dict1 = {}
        main = {}
        f = []
        f1 = []
        f2 = []
        histor = []
        box = []
        lbox = []
        boxdic = {}
        if len(idxs) > 0:
            print(len(idxs))
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                lbox = [x, y, w, h]
                box.append(list(lbox))
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # print((crop_img))
                bb1 = (w * h) / (image.shape[0] * image.shape[1])
                bb.append(bb1)
                lab.append(LABELS[classIDs[i]])
                conf.append(confidences[i])
                area = dict(zip(bb, lab))
                dic = dict(zip(conf, lab))
                dic1 = dict(zip(bb, lab))
            for k1 in range(0, len(idxs)):
                boxdic[k1] = [lab[k1], box[k1]]
            for k in range(0, len(lab)):
                x = abs(box[k][0])
                y = abs(box[k][1])
                w = abs(box[k][2])
                h = abs(box[k][3])
                crop_img = image[y:y + h, x:x + w]
                src = crop_img
                r = src[:, :, 0]
                out = np.divide(r, 4)
                z = np.round(out)
                rz = z.astype(np.int64)
                np.histogram(rz.flatten(), bins=l)
                hist, bins = np.histogram(rz.flatten(), bins=l)
                # histor.append(list(hist))
                r1 = src[:, :, 1]
                out1 = np.divide(r1, 4)
                z1 = np.round(out1)
                rz1 = z1.astype(np.int64)
                np.histogram(rz1.flatten(), bins=l)
                hist1, bins = np.histogram(rz1.flatten(), bins=l)
                # histog.append(list(hist1))
                r2 = src[:, :, 2]
                out2 = np.divide(r2, 4)
                z2 = np.round(out2)
                rz2 = z2.astype(np.int64)
                np.histogram(rz2.flatten(), bins=l)
                hist2, bins = np.histogram(z2.flatten(), bins=l)
                # histob.append(list(hist2))python yolo.py  --yolo yolo-coco
                dict1[k] = [lab[k],list(hist), list(hist1), list(hist2)]
        #print(dict1)
    # if (len(dict1)==len(lab)):
    # l = []

    # print(histor)
    # print(histog)
    # print(histob)
    # print(boxdic)
    # for i in boxdic.keys():
    # boxdic[i]

        dictionary = dict(zip(bb, zip(lab, conf)))
        print("MODIFED WITH 50% PROBALITY ,WILL SHOW CLASS NAME W.R.T COUNT--custom")
        print(str(img))
        for j in range(0, len(lab)):
            count = 0
            a = 0
            for m in area:
                if lab[j] == area[m]:
                    count += 1
                    a = a + m
                    if (a >= 1.0):
                        a = 1
            main[lab[j]] = [round(a,1), count]
        print(main)


    # from scipy.spatial import distance
    # distance.euclidean([1, 0, 0], [0, 1, 0])
        fname = str(img)
        V = fname
        V = str(main) + "#" + str(dict1) + "#" + fname
        f = open("features.txt", "a")
        f.write(V + " \n")
        f.close()
        f = open("features.txt", "r")
        for r in f:
            l1 = list(r.split('*'))
        f.close()
    if (len(main) == 0):
        print("cannot")
        #import shutil
        #shutil.move(str(img), "C:\\Users\\tanka\\yolo-object-detection\\not_working")
# shutil.move(str(img), "C:\\Users\\tanka\\yolo-object-detection\\corel_working")
#print("-------------------------------------------------------------------------------------")

# show the output imag
cv2.imshow("Image", image)
cv2.waitKey(0)