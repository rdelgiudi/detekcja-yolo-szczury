import cv2
import numpy as np
import time
import os
import sys
from numpy.core.shape_base import atleast_2d
import math
from datetime import datetime

stopSignal = False

class Logger(object):
    def __init__(self, videoname):
        self.terminal = sys.stdout
        logName = os.path.basename(videoname)
        logName = os.path.splitext(logName)[0]
        logName += ".log"
        logDir = "output/" + logName
        self.log = open(logDir, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

def inCorner(x1, y1, width, height):
    dw = round(width / 2.75)
    dh = round(height / 2.75)
    corner1 = [(0, 0), (dw, dh)]
    corner2 = [(width-dw, 0), (width, dh)]
    corner3 = [(0, height - dh), (dw, height)]
    corner4 = [(width-dw, dh), (width, height)]
    if (x1 >= corner1[0][0] and x1 <= corner1[1][0] and y1 >= corner1[0][1] and y1 <= corner1[1][1]):
        return True, 1
    elif (x1 >= corner2[0][0] and x1 <= corner2[1][0] and y1 >= corner2[0][1] and y1 <= corner2[1][1]):
        return True, 2
    elif (x1 >= corner3[0][0] and x1 <= corner3[1][0] and y1 >= corner3[0][1] and y1 <= corner3[1][1]):
        return True, 3
    elif (x1 >= corner4[0][0] and x1 <= corner4[1][0] and y1 >= corner4[0][1] and y1 <= corner4[1][1]):
        return True, 4
    else:
        return False, -1


def getRectCenter(x1, y1, x2, y2):
    xCenter = (x1 + x2) / 2
    yCenter = (y1 + y2) / 2
    return xCenter, yCenter

def calcDist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def vectorProduct(x1, y1, x2, y2, x3, y3):
    X1 = x3 - x1
    Y1 = y3 - y1
    X2 = x2 - x1
    Y2 = y2 - y1

    return (X1 * Y2) - (X2 * Y1)

def checkExtremes(x1, y1, x2, y2, x3, y3):
    return min(x1, x2) <= x3 and x3 <= max(x1, x2) and min(y1, y2) <= y3 and y3 <= max(y1, y2)

def startDetect(videoname, vs ,conf : float, thold : float, outputfile : str, fileonly : bool, drawpaths : bool, maxloss : int, logging : bool,calccross : bool, pixelspercm : float, isQt : bool, bar):
    if isQt:
        import progresscontrollerqt as progresscontroller
    else:
        import progresscontroller

    programStart = time.time()
    if fileonly is True and outputfile is None:
        print("[BŁĄD] Za mało argumentów: Podaj nazwę pliku wynikowego.")
        exit(1)

    #wczytanie listy wykrywalnych obiektów z pliku coco.names
    objectNames = ["rat"]

    #generacja zestawu kolorów dla każdego obiektu (maks 100)
    np.random.seed(981119)
    colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

    #zaznaczenie ścieżki plików dla algorytmu YOLOv3
    weightsPath = "yolov3_rats_v3_1.weights"
    configPath = "yolov3_testing.cfg"

    #wczytanie detektora YOLOv3 z wytrenowanego zestawu danych COCO oraz uruchomienie CUDA (jeżeli jest wspierane)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

     #Uzyskanie danych wczytanego obrazu
    fwidth = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ffps = vs.get(cv2.CAP_PROP_FPS)
    totalframes = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    #przeskalowanie klatki, zależnie od rozmiaru
    if fwidth >= 2560 and fheight >= 1440:
        scale_percent = 1/4
    elif fwidth >= 1366 and fheight >= 768:
        scale_percent = 1/2
    else:
        scale_percent = 1
    width = int(fwidth * scale_percent)
    height = int(fheight * scale_percent)
    dim = (width, height)

    #jeżeli zapisujemy do pliku, to tutaj przygotowywany jest strumień wyjściowy
    if outputfile is not None:
        if not os.path.isdir("output"):
            os.mkdir("output")
        outputname = outputfile
        outputfile = "output/" + outputfile + ".mp4"
        if fileonly is False:
            out = cv2.VideoWriter(outputfile,cv2.VideoWriter_fourcc('a','v','c','1'), ffps, dim)
        else:
            out = cv2.VideoWriter(outputfile,cv2.VideoWriter_fourcc('a','v','c','1'), ffps, (fwidth, fheight))

    #przygotowanie wymiarów okna do optymalnego oglądania wyników
    if not fileonly:
        window_width = int(fwidth * scale_percent)
        window_height = int(fheight * scale_percent)
        cv2.namedWindow("Detekcja wideo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detekcja wideo", window_width, window_height)

    #przygotowanie zmiennych potrzebnych do progressbar oraz systemu ID
    #oraz pomiaru fps
    frametimes = []
    previousIDs = []
    previousCoord = []
    paths = []
    vectors = []
    speed = []
    speeds = []

    counter = []
    pathcounter = []
    successfulFrames = []
    cornerLT = []
    cornerRT = []
    cornerLB = []
    cornerRB = []

    for i in range(100):
        paths.append([])
        speed.append(0)
        speeds.append([])
        vectors.append([0, 0])
        counter.append(0)
        pathcounter.append(0)
        successfulFrames.append(0)
        cornerLT.append(0)
        cornerRT.append(0)
        cornerLB.append(0)
        cornerRB.append(0)

    print("[INFO] Przygotowania zakończone. Rozpoczynanie detekcji...")

    while True:
        start = time.time()
        ret, frame = vs.read()
        #jeżeli strumień wideo zakończył się, to przerywane jest działanie programu
        if ret == False:
            break
        
        framenum = int(vs.get(cv2.CAP_PROP_POS_FRAMES))

        if stopSignal:
            totalframes = framenum - 1
            break

        #utworzenie progressbar, wartość maksymalna wynosi maksymalną liczbę klatek
        if framenum == 1:
            if 'out' in locals():
                print("[INFO] Zapisywanie pliku...")
            
            bar = progresscontroller.setupBar(bar, totalframes)
                
        
        #uzyskanie wymiarów klatki
        (H, W) = frame.shape[:2]

        #uzyskanie nazw warstw na wyjściu algorytmu
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        #stworzenie blob i przekazanie do YOLO, który zwróci nam bounding box (warstwy) oraz prawdopodobieństwa
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        #inicjalizacja list wykrytych bounding box, prawdopodbieństw oraz nazw obiektów
        boxes = []
        confidences = []
        classIDs = []

        #sortowanie wyników, odfilrtowanie słabych wyników
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > conf:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height), centerX, centerY])
                    confidences.append(float(confidence))
                    classIDs.append(classID)                        

        #uruchomienie Non Maximum Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, thold)

        #inicjalizacja zmiennych potrzebnych do systemu ID
        newCoord = []
        IDs = []

        #przetworzenie wyników z NMS oraz narysowanie ich na klatce
        if len(idxs) > 0:
            for iterator, i in enumerate(idxs.flatten()):
                if iterator > 1:
                    break
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                (centerX, centerY) = (boxes[i][4], boxes[i][5])

                #system ID przydziela ID w pierwszej klatce w kolejności od największej do najmniejszej pewności
                #po czym w kolejnych klatkach ID jest utrzymywane na podstawie najmniejszej odległości
                #od ostatniej lokacji
                if not previousIDs or framenum == 1:
                    IDs.append(iterator)
                    previousCoord.append([centerX, centerY])
                else:
                    distances = []
                    for coord in previousCoord:
                        distances.append(calcDist(coord[0], coord[1], x, y))
                    idIndex = distances.index(min(distances))
                    newID = previousIDs[idIndex]
                    #ewentualne powstałe duplikaty są eliminowane i przydzielany jest im najmniejszy
                    #możliwy ID
                    if len(IDs) > 0:
                        duplicateDetected = True
                        oldz = False
                        while duplicateDetected:
                            duplicateDetected = False
                            for idval in IDs:
                                if idval == newID:
                                    if not oldz:
                                        newID = 0
                                        duplicateDetected = True
                                        oldz = True
                                        break
                                    newID += 1
                                    duplicateDetected = True
                                    break
                    IDs.append(newID)
                    newCoord.append([x, y])
                    confidence = round(confidences[i] * 100)
                
                if IDs[iterator] in previousIDs  and IDs.index(IDs[iterator]) < len(previousCoord):
                    vectors[IDs[iterator]] = [newCoord[IDs.index(IDs[iterator])][0] - previousCoord[previousIDs.index(IDs[iterator])][0], 
                    newCoord[IDs.index(IDs[iterator])][1] - previousCoord[previousIDs.index(IDs[iterator])][1]]

                paths[IDs[iterator]].append([centerX, centerY])

                if len(paths[IDs[iterator]]) >= 30:
                    tempdist = []
                    prevelem = 0
                    for j, pathi in enumerate(reversed(paths[IDs[iterator]])):
                        if not j:
                            prevelem = pathi
                            continue

                        tempdist.append(calcDist(prevelem[0], prevelem[1], pathi[0], pathi[1]))
                        prevelem = pathi

                        if j >= ffps: 
                            break
                    
                    if pixelspercm:
                        tempspeed = sum(tempdist) / pixelspercm 
                    else:
                        tempspeed = sum(tempdist)
                    
                    speed[IDs[iterator]] = tempspeed
                    speeds[IDs[iterator]].append(tempspeed)
                    

                ifCorner, whichCorner = inCorner(centerX, centerY, fwidth, fheight)
                if ifCorner:
                    if whichCorner == 1:
                        cornerLT[IDs[iterator]] += 1
                    elif whichCorner == 2:
                        cornerRT[IDs[iterator]] += 1
                    elif whichCorner == 3:
                        cornerLB[IDs[iterator]] += 1
                    elif whichCorner == 4:
                        cornerRB[IDs[iterator]] += 1

                if not IDs:
                    color = [0, 0, 255]
                    cv2.rectangle(frame, (x,y), (x + w, y + h), color, 2)
                    text = "ID:{} {} c: {}%".format("?",objectNames[classIDs[i]], confidence)
                    textLen = len(text) * 9
                    cv2.rectangle(frame, (x,y), (x + textLen, y - 20), color, -1)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
                else:
                    color = [int(c) for c in colors[IDs[iterator]]]
                    cv2.rectangle(frame, (x,y), (x + w, y + h), color, 2)
                    text = "ID:{} {} c: {}%".format(IDs[iterator],objectNames[classIDs[i]], confidence)
                    textLen = len(text) * 9
                    cv2.rectangle(frame, (x,y), (x + textLen, y - 20), color, -1)
                    cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 2)
                
                successfulFrames[IDs[iterator]] += 1

        # w8 = round(fwidth/2.75)
        # h8 = round(fheight/2.75)
        # cv2.rectangle(frame, (0, 0), (w8, h8), [255, 0, 0], -1)
        # cv2.rectangle(frame, (0, fheight), (w8, fheight - h8), [255, 0, 0], -1)
        # cv2.rectangle(frame, (fwidth, 0), (fwidth - w8, h8), [255, 0, 0], -1)
        # cv2.rectangle(frame, (fwidth, fheight), (fwidth - w8, fheight - h8), [255, 0, 0], -1)
        #rozbudowanie systemu ID, jeżeli nie zostanie wykryte ID to jego pozycja zostanie zaktualizowana na podstawie ostatniego znanego wektora, aż do 5 razy
        #po czym ID się przedawnia i jego pozycja zostaje usunięta
        for i ,prevID in enumerate(previousIDs):
            if not prevID in IDs and counter[prevID] < 5:
                IDs.append(prevID)
                newCoord.append([previousCoord[i][0] + vectors[prevID][0], previousCoord[i][1] + vectors[prevID][1]])
                counter[prevID] += 1
            else:
                counter[prevID] = 0

        if drawpaths:
            for i, path in enumerate(paths):
                if i not in IDs:
                    pathcounter[i] += 1
                else:
                    pathcounter[i] = 0
                if pathcounter[i] > maxloss:
                    pathcounter[i] = 0
                    paths[i] = []
                if path != [] and len(path) > 1:
                    color = [int(c) for c in colors[i]]
                    npaths = np.asarray(path, dtype=np.int32)
                    cv2.polylines(frame, [npaths], False, color, 2)

        framecopy = frame.copy()
        
        iter = 99
        for reviter, sp in enumerate(reversed(speed)):
            if iter in IDs and iter <= 1:
                if pixelspercm:
                    speedtext = "ID: {} Speed: {:.2f} cm/s".format(iter, sp)
                else:
                    speedtext = "ID: {} Speed: {:.2f} pixel/s".format(iter, sp)
                color = [int(c) for c in colors[iter]]
                cv2.putText(frame, speedtext, (10, fheight - round(fheight * 0.01) - ((reviter*30) - (98 * 30))),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            iter -= 1

        if not IDs:
            IDs = previousIDs.copy()
        if not newCoord:
            newCoord = previousCoord.copy()
        previousIDs = IDs.copy()
        previousCoord = newCoord.copy()
        
        #ostatnie przygotowania, pomiar frametime oraz fps, oraz ich umieszczenie na
        #klatce obrazu (oraz ewentualny zapis do pliku, jeżeli opcja jest aktywna)
        end = time.time()

        if outputfile is not None:
            out.write(frame)

        framet = float(end- start)
        frametimes.append(framet)
        if (len(frametimes) > 10):
            del frametimes[0]
        fps = 0.0
        for frameti in frametimes:
            fps += frameti
        fps = fps/ len(frametimes)
        frametime = "Frametime: {:.1f} ms".format(fps * 1000)
        fps = 1/fps
        fpsinfo = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fpsinfo, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, frametime, (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if fileonly is False:
            cv2.imshow("Detekcja wideo", frame)

        progresscontroller.updateBar(bar, framenum)
        key = cv2.waitKey(1)
        if key == ord("e"):
            totalframes = framenum
            break
    

    print("")
    pathcrossed = 0
    cv2.destroyAllWindows()

    if outputfile is not None:
        framePath = "output/" + outputname + "LastFrame.jpg"
        cv2.imwrite(framePath, framecopy)

    if calccross:
        print("[INFO] Rozpoczynanie analizy liczby przecięć ścieżek, proszę czekać...")
        bar = progresscontroller.setupBar(bar, (len(paths[0]) -1))
        for j in range(0, len(paths[0]) - 1):
            progresscontroller.updateBar(bar, j)
            for i in range(0, len(paths[1]) - 1):
                pt1 = paths[1][i]
                pt2 = paths[1][i+1]
                pt3 = paths[0][j]
                pt4 = paths[0][j+1]

                x1, x2, x3, x4 = pt1[0], pt2[0], pt3[0], pt4[0]
                y1, y2, y3, y4 = pt1[1], pt2[1], pt3[1], pt4[1]

                v1 = vectorProduct(x3, y3, x4, y4, x1, y1)
                v2 = vectorProduct(x3, y3, x4, y4, x2, y2)
                v3 = vectorProduct(x1, y1, x2, y2, x3, y3)
                v4 = vectorProduct(x1, y1, x2, y2, x4, y4)

                if (((v1 > 0 and v2 < 0) or (v1 < 0 and v2 > 0)) and ((v3 > 0 and v4 < 0) or (v3 < 0 and v4 > 0))):
                    pathcrossed += 1
                    #i += ffps * 3
                    continue

                if(v1 == 0 and checkExtremes(x3, y3, x4, y4, x1, y1)):
                    pathcrossed += 1
                    #i += ffps * 3
                    continue
                if(v2 == 0 and checkExtremes(x3, y3, x4, y4, x2, y2)):
                    pathcrossed += 1
                    #i += ffps * 3
                    continue
                if(v3 == 0 and checkExtremes(x1, y1, x2, y2, x3, y3)):
                    pathcrossed += 1
                    #i += ffps * 3
                    continue
                if(v4 == 0 and checkExtremes(x1, y1, x2, y2, x4, y4)):
                    pathcrossed += 1
                    #i += ffps * 3
                    continue
    if logging:
        if not os.path.isdir("output"):
            os.mkdir("output")
        stdoutcopy = sys.stdout
        if outputfile is not None:
            sys.stdout = Logger(outputname)
        else:
            sys.stdout = Logger(videoname)
    print("")
    print("Data i czas: {}".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    print("")
    print("////////////////////////////////////////////////////////////////////////////////////////")
    print("                           Statystki przebadanego filmu")
    print("////////////////////////////////////////////////////////////////////////////////////////")
    print("")


    print("Minimalny próg detekcji: {:.2f}".format(conf))
    print("Próg NMS: {:.2f}".format(thold))
    print("Liczba przebadanych klatek: {} ({:.2f} sekund)".format(framenum, framenum / ffps))
    for i in range(0, 99):
        if successfulFrames[i] != 0:
            successpercent = float((successfulFrames[i]/totalframes) * 100)
            lostframes = int(totalframes - successfulFrames[i])
            print("Obiekt nr {} wyśledzony przez {:.2f}% klatek (zgubiony w {} klatkach).".format(i, successpercent, lostframes))

    if calccross:
        print("Obiekt nr 0 i 1 przecieły drogi {} razy.".format(pathcrossed))
    
    print("\n")
    print("Przebycie w rogach:")
    for i in range(0, 99):
        if cornerLB[i] != 0 or cornerLT[i] != 0 or cornerRB[i] != 0 or cornerRT[i] != 0:
            print("")
            print("Obiekt {}:".format(i))
            print("Lewy górny róg: {} klatek ({:.2f} sekund)".format(cornerLT[i], cornerLT[i] / ffps))
            print("Prawy górny róg: {} klatek ({:.2f} sekund)".format(cornerRT[i], cornerRT[i] / ffps))
            print("Lewy dolny róg: {} klatek ({:.2f} sekund)".format(cornerLB[i], cornerLB[i] / ffps))
            print("Prawy dolny róg: {} klatek ({:.2f} sekund)".format(cornerRB[i], cornerRB[i] / ffps))

    print("\n")
    print("Droga i prędkość:")
    for i in range(0, 99):
        if successfulFrames[i] != 0:
            objdist = []
            for j, point in enumerate(paths[i]):
                if not j:
                    prevx = point[0]
                    prevy = point[1]
                    continue
                x = point[0]
                y = point[1]
                objdist.append(calcDist(x, y, prevx, prevy))

                prevx = x
                prevy = y


            print("")
            print("Obiekt {}:".format(i))
            if not pixelspercm:
                print("Przebyta droga: {:.2f} pikseli.".format(sum(objdist)))
                print("Średnia prędkość: {:.2f} pikseli/s.".format(np.mean(speeds[i])))
            else:
                print("Przebyta droga: {:.2f} cm.".format(sum(objdist) / pixelspercm))
                print("Średnia prędkość: {:.2f} cm/s.".format(np.mean(speeds[i])))
            

    print("////////////////////////////////////////////////////////////////////////////////////////")
    print("")
    programEnd = time.time()
    print("[INFO] Działanie programu trwało {:.2f} sekund.".format(programEnd - programStart))
    
    if logging:
        sys.stdout = stdoutcopy

    vs.release()
    if 'out' in locals():
        out.release()