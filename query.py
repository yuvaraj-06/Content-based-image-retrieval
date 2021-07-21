# USAGE
# python query.py  --yolo yolo-coco
# import the necessary packages
import numpy as np
import argparse
import glob
import time
import cv2
import os
import shutil

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
def file_hand(main, dict1, lab):
    def search_multiple_strings_in_file(file_name, list_of_strings):
        line_number = 0
        list_of_results = []
        with open(file_name, 'r') as read_obj:
            with open("C:\\Users\\tanka\\yolo-object-detection\\features1.txt", "w") as f1:
                for line in read_obj:
                    line_number += 1
                    for string_to_search in list_of_strings:
                        if string_to_search in line:
                            list_of_results.append((string_to_search, line_number, line.rstrip()))
                            f1.write(line.rstrip() + " \n")
        return list_of_results

    matched_lines = search_multiple_strings_in_file("C:\\Users\\tanka\\yolo-object-detection\\features.txt", lab)
    fdic = {}
    results1 = []
    eres = []
    import ast

    def ed(v1, v2):
        return sum((p - q) ** 2 for p, q in zip(v1, v2)) ** .5

    with open("C:\\Users\\tanka\\yolo-object-detection\\features1.txt", "r") as f1:
        for line in f1:
            a = line.split("#")
            a1 = ast.literal_eval(a[0])
            for k in main:
                for k1 in a1:
                    if k == k1:
                        results1.append(a[len(a) - 1])
                        if main[k][1] == a1[k][1]:
                            # print(a[len(a)-1])
                            # a2 = ast.literal_eval(a[0])
                            if (((main[k][0] + 0.3) >= a1[k][0]) and ((main[k][0] - 0.3) <= a1[k][0])):

                                #print("CHECKPOINT 1")
                                # print(a[len(a) - 1])
                                # print("RESULTS OF IMAGES W.R.T COUNT & AREA & LABELS")
                                a2 = ast.literal_eval(a[1])
                                for zfi in dict1:
                                    # print(z)
                                    # print(dict1[z][0])
                                    # print(a2[z][0])
                                    if len(dict1)==len(a2):

                                        if dict1[zfi][0] == a2[zfi][0]:
                                            #print("CHECKPOINT 2")
                                            sum1 = 0
                                            for p in range(1, 4):
                                                s = ((ed(dict1[zfi][p], a2[zfi][p])))
                                                # print(s)
                                                sum1 = sum1 + s
                                                path = a[len(a) - 1]
                                                path = path.rstrip('\n')
                                                path = path.rstrip(' ')
                                            t1 = round(sum1 / 3, 1)
                                            fdic[path] = t1
                                        else:
                                            continue
        f1dic = ({k: v for k, v in sorted(fdic.items(), key=lambda item: item[1])})
        results = list((f1dic.keys()))
        res = []
        res1 = []
    for i in results1:
        i = i.rstrip('\n')
        i = i.rstrip(' ')
        res1.append(i)
    res1 = list(set(res1))

    def diff(list1, list2):
        out = []
        for ele in list1:
            if not ele in list2:
                out.append(ele)
        return out

    fes = []
    print("ALMOST DONE")
    if (len(results) < 10):
        fres = (res1) + ((results))
        fres1 = diff(fres, results)
        # print(fres1)
        c = 0
        for i in fres1:
            results.insert(len(results) + c, str(i))
            c += 1
        c1 = 0
        for i in results:
            shutil.copy(str(i), "C:\\Users\\tanka\\yolo-object-detection\\result")
            newname = os.path.join("C:\\Users\\tanka\\yolo-object-detection\\result", str(c1) + ".jpg")
            lname = str(i).split('\\')[-1:][0]
            os.rename("C:\\Users\\tanka\\yolo-object-detection\\result" + "\\" + lname, newname)
            c1 += 1
            if (c1 == 10):
                print("THANKS FOR USING")
                break
    else:
        c1 = 0
        for i in results:
            shutil.copy(str(i), "C:\\Users\\tanka\\yolo-object-detection\\result")
            newname = os.path.join("C:\\Users\\tanka\\yolo-object-detection\\result", str(c1) + ".jpg")
            lname = str(i).split('\\')[-1:][0]
            os.rename("C:\\Users\\tanka\\yolo-object-detection\\result" + "\\" + lname, newname)
            c1 += 1
            if (c1 == 10):
                print("THANKS FOR USING")
                break
for img in glob.glob("C:\\Users\\tanka\\yolo-object-detection\\query\\*.jpg"):
#for img in glob.glob("C:\\Users\\tanka\\yolo-object-detection\\images\\*.jpg"):
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

    histor = []
    box=[]
    lbox=[]
    boxdic={}

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
                crop_img = image[y:y + h, x:x + w]
                src = crop_img
                #histr = cv2.calcHist([src], [0], None, [256], [0, 256])
                #histg = cv2.calcHist([src], [1], None, [256], [0, 256])
                #histb = cv2.calcHist([src], [2], None, [256], [0, 256])
                r= src[:, :, 1]
                out = np.divide(r, 4)
                z =np.round(out)
                rz=z.astype(np.int64)
                np.histogram(rz.flatten(), bins=l)
                hist, bins = np.histogram(rz.flatten(), bins=l)
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
        dictionary = dict(zip(bb, zip(lab, conf)))
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
        file_hand(main,dict1,lab)
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
        histor = []
        box = []
        lbox = []
        boxdic = {}
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                lbox = [x, y, w, h]
                box.append(list(lbox))
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
        dictionary = dict(zip(bb, zip(lab, conf)))
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
    file_hand(main, dict1, lab)



