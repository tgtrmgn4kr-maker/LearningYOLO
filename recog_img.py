import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
from name_dictionary import class_dict

model = YOLO("yolo12n.pt")

img = cv2.imread('img/face.png') # Read the image with OpenCV and convert it into a BGR numpy array

objs = model.predict(img, 
                     conf=0.4, 
                     save=False, 
                     show=False, 
                     classes=[0,1,2,3,5,6,7]
                     ) # Detect specific types in coco.yaml 

for obj in objs: # images
    boxes = obj.boxes 
    for box in boxes: # Box which surrounding the detected object

        x1, y1, x2, y2 = box.xyxy[0] # Get the coordinates of the objects (Tensor format)

        x1, y1, x2, y2 = int (x1), int (y1), int (x2), int (y2) # Convert into integer

        class_index = int (box.cls[0]) # box.cls[0] is float

        name = class_dict[str(class_index)]

        confidence = round(float(box.conf[0]), 2)

        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # OpenCV uses BGR, Pillow uses RGB

        label = name +" : "+ str(confidence)

        draw = ImageDraw.Draw(pil_image)
        
        draw.rectangle([x1, y1, x2, y2], 
                       outline='orange', 
                       width=2) # Draw boxes with Pillow
        
        _, _, txt_w, txt_h = draw.textbbox((0,0),label) # Calculate how large space the label need

        top = (x1, y1-txt_w+28)
        bottom = (x1+txt_w+4, y1)
        draw.rectangle([top,bottom],
                       fill='orange') # Draw the text box
        
        draw.text((x1+2,y1-txt_h-2),
                  label,
                  fill='black') # Draw the text
        
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) # Convert the array back from Pillow to OpenCV

cv2.imshow("Photo", img)
while cv2.getWindowProperty("Photo", cv2.WND_PROP_VISIBLE) >= 1: # If the window is opened 
    if cv2.waitKey(1) & 0b11111111 == ord('q'):
        break






        


        




        






































