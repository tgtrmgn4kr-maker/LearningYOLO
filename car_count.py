import cv2
import time 
from name_dictionary import class_dict
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

cap = cv2.VideoCapture("videos/cars.mp4")
assert cap.isOpened, "Loading video failed."

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Video width: {w}, height: {h}, fps:{fps}")

output_video = cv2.VideoWriter("output.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h)
                               )
model = YOLO("yolo12n.pt")

names = class_dict
current_time = 0
last_update_time = 0

while True:
    ret, frame = cap.read()
    if ret:
        objs = model.predict(frame,
                             conf=0.35,
                             classes=[2, 5, 7]
                             )
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(frame_pil)

        for obj in objs:
            boxes = obj.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                draw.rectangle([x1, y1, x2, y2],
                               outline="orange",
                               width=2
                               ) # Draw the box
                
                name = class_dict[str(int(box.cls[0]))]

                conf = round(box.conf[0].item(), 2)

                label = f"{name} : {conf}"

                _, _, txt_w, txt_h = draw.textbbox((0,0),label)

                top = (x1, y1-txt_w+28)
                bottom = (x1+txt_w+4, y1)
                draw.rectangle([top,bottom],
                                fill='orange'
                                ) # Draw the text box 
                
                draw.text((x1+2,y1-txt_h-2),
                           label,
                           fill='black') # Draw the text
                
        frame_output = cv2.cvtColor(np.array(frame_pil),cv2.COLOR_RGB2BGR)

        current_time = time.time()
        fps = 1 / (current_time - last_update_time)
        last_update_time = current_time
        cv2.putText(frame_output,             # image source
                    f"FPS : {int(fps)}",      # text
                    (30, 680),                # text coordinate
                    cv2.FONT_HERSHEY_PLAIN,   # font
                    3,                        # font size
                    (255, 255, 255),          # font color
                    3                         # font thickness
                    )
        
        output_video.write(frame_output)

        cv2.imshow("Video", frame_output)

        if cv2.waitKey(1) & 0b1111_1111 == ord('q'):
            break
    else:
        break
cap.release()
output_video.release()
cv2.destroyAllWindows()

































































