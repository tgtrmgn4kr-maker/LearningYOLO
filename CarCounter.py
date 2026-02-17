import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO

model = YOLO("yolo12n.pt")

class CarCounter:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.line_v = int(w/2)
        self.line_h = int(h/2)
        self.right_counter = 0
        self.left_counter = 0
        self.prev_centers = {}      # id -> (cx, cy)
        self.counted_ids = set()    # tracks already counted

    def update(self, id, box):
        if id is None:
            return

        x1, y1, x2, y2 = box
        car_x = int(x1 + (x2 - x1) / 2)
        car_y = int(y1 + (y2 - y1) / 2)

        prev = self.prev_centers.get(id)
        if prev is not None and id not in self.counted_ids: 
            prev_x, prev_y = prev
            if prev_y < self.line_h <= car_y and car_x > self.line_v: # if objects cross the middle line
                self.right_counter += 1
                self.counted_ids.add(id) # add the id in the set if it is not recorded
            elif prev_y > self.line_h >= car_y and car_x < self.line_v: 
                self.left_counter += 1
                self.counted_ids.add(id) 

        self.prev_centers[id] = (car_x, car_y) # update the new coordinate of the objects

    def draw_counter(self, frame): # draw the line to detect if objects crossed it
        cv2.line(frame,
                 (0, self.line_h),
                 (self.w, self.line_h),
                 (0, 0, 255),
                 2
                 )
        
        cv2.line(frame,
                 (self.line_v, 0),
                 (self.line_v, self.h),
                 (255, 0, 0),
                 2
                 )

        cv2.putText(frame,
                    f"COUNT : {self.left_counter}",
                    (30, self.h-50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                    )
        
        cv2.putText(frame,
                    f"COUNT : {self.right_counter}",
                    (self.line_v+30, self.h-50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                    )
        
def draw_label(frame, id, box): # draw rectangle and label
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    color = COLORS[id % 6]

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(frame,
                  (x1, y1),
                  (x2, y2),
                  color,
                  2)
    
    cv2.putText(frame,
                f"ID : {id}",
                (x1, y1-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
                )
        
def main():
    cap = cv2.VideoCapture("videos/tokyo_720p.mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_video = cv2.VideoWriter("car_counter.mp4",
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   frame_fps,
                                   (frame_width, frame_height)
                                   )
    
    car_counter = CarCounter(frame_width, frame_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No such video")
            break

        results = model.track(frame,
                              persist=True,
                              conf = 0.3,
                              classes = [2, 5, 7]
                              ) # objects detected  in frames
        
        for box in results[0].boxes:
            r = box.xyxy.tolist() # coordinate
            if box.id is None:
                continue
            id = int(box.id) # get ids of objects
            car_counter.update(id, r[0])
            draw_label(frame, id, r[0])

        car_counter.draw_counter(frame)
        output_video.write(frame)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0b1111_1111 == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()











































