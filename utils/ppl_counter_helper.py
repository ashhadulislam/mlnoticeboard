import os
import cv2
import wget
import numpy as np



# def download_weights():

#     # Load Yolo
#     print("DownLOADING YOLO")
#     weights_url="https://pjreddie.com/media/files/yolov3.weights"
    
#     yolocation=os.path.join(os.getcwd(),"static","yolo_ppl_counter")
#     wt_output_directory=os.path.join(yolocation)
#     if not os.path.isfile(os.path.join(yolocation,"yolov3.weights")):    
#         print("FIle doesnt exist in download weights, downloading")
#         wts_filename = wget.download(weights_url,out=wt_output_directory)
#         print("Weight received")

#     else:
#         print("FIle exists in download weights, not downloading")
#         wts_filename="yolov3.weights"

def setupYOLO():
    # Load Yolo
    print("LOADING YOLO")
    # weights_url="https://pjreddie.com/media/files/yolov3.weights"
    
    yolocation=os.path.join(os.getcwd(),"static","yolo_ppl_counter")
    # wt_output_directory=os.path.join(yolocation)
    # if not os.path.isfile(os.path.join(yolocation,"yolov3.weights")):    
    #     print("FIle doesnt exist, downloading")
    #     wts_filename = wget.download(weights_url,out=wt_output_directory)

    # else:
    #     print("FIle exists, not downloading")
    #     wts_filename="yolov3.weights"
    wts_filename="MobileNetV2-YOLOv3-Lite-coco.weights"
    config_file_name="MobileNetV2-YOLOv3-Lite-coco.cfg"
    net = cv2.dnn.readNet(os.path.join(yolocation,wts_filename)
        , os.path.join(yolocation,config_file_name))

    #save all the names in file o the list classes
    classes = []
    with open(os.path.join(yolocation,"coco.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    #get layers of the network
    layer_names = net.getLayerNames()
    #Determine the output layer names from the YOLO model 
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("YOLO LOADED")  
    return net,layer_names,output_layers,classes  



def detect_persons(frame,net,layer_names,output_layers,classes):

    detection_confidence=0.75
    print("In detect_persons",frame.shape)

    h=frame.shape[0]
    w=frame.shape[1]
    x_start=0
    y_start=0


    

    

    startX=x_start
    startY=y_start
    endX=startX+w
    endY=startY+h


    # cv2.rectangle(frame, (startX, startY), (endX, endY), (0,250,50), 2)
        
    roi=frame[startY:endY, startX:endX]
    print("ROI shape = ",roi.shape)
    # (h, w) = frame.shape[:2]
    (height, width) = roi.shape[:2]
    channels=roi.shape[2]

    # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
    # blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 0.007843, (300, 300), 127.5)
    # blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 0.007843, (300, 300), 127.5)
    blob = cv2.dnn.blobFromImage(roi, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    
    # prototxt=args["prototxt"]
    # ppl_model=args["model"]

    net.setInput(blob)
    outs = net.forward(output_layers)
    person_class_id=0
    class_ids = []
    confidences = []
    boxes = []

    
    # Focal length
    F = 615


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > detection_confidence:
                if class_id==person_class_id:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)                    
    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    print("Count of people = ",len(indexes))

    count_people=0
    people_dict={}
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # label = str(classes[class_ids[i]])+" "+str(round(confidences[i]*100))
            # color = colors[class_ids[i]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            startX, endX, startY, endY=x,x+w,y,y+h
            
            height=round(endY-startY,4)
            # Distance from camera based on triangle similarity
            distance=(165*F)/height

            # Mid point of bounding box
            x_mid = round((startX+endX)/2,4)
            y_mid = round((startY+endY)/2,4)

            # Mid-point of bounding boxes (in cm) based on triangle similarity technique
            x_mid_cm = (x_mid * distance) / F
            y_mid_cm = (y_mid * distance) / F


            # print(startX, startY, endX, endY)
            # cv2.putText(frame, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,1/2, color, 2)
            people_dict[count_people]={}
            people_dict[count_people]["coords"]=(startX, startY, endX, endY)
            people_dict[count_people]["confidence"]=confidences[i]
            people_dict[count_people]["position"]=(x_mid_cm,y_mid_cm,distance)
            count_people+=1
                
    return frame,count_people,people_dict
