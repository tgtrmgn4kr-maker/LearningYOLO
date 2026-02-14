from ultralytics import YOLO

model = YOLO('yolo12n.pt')

img = 'img/vehicles.jpg'

results = model.predict(img, 
                    conf=0.25,
                    save=True,
                    show=True
                    )  
'''
results contain all the information of the image
'''
print(results[0].boxes[0].xyxy)
'''
boxes is a list contains information of all objects detected in the image
xyxy is one attribute of boxes recording the coordinates of a object
'''