LearningYOLO
===
This repository is for recording my learning progress

## What I practiced
- Using the model ***yolo12n.pt*** to detect the objects in the image
- Understanding what is contained in `results.boxes` of YOLO model 
- Using `OpenCV` to read the image with OpenCV and convert it to a BGR numpy array
- Using `Pillow` to draw lines and boxes on images and frames in a video
- Put a dictionary in an additional file and call it
- Using a class `CarCounter` to manage parameters in `CarCounter.py`

## Notes
- `OpenCV` and `Pillow` use different arrays to store color files. `OpenCV` uses BGR
 and `Pillow` uses RGB so I should convert it several times if I have to use them both

## Files and Directories

### files
- `test.py` : understand what is contained in `results.boxes` of YOLO model
- `recog_img.py` and `recog_img2.py` : to identify if cars exist in the image
- `name_dictionary.py` : store dictionary of names in coco.yaml
- `CarCounter.py` : to count cars which crossed the line

### Models
- `yolo12n.pt`
- `yolo12l.pt`

### Directories
- `img` : to store images for practicing
- `videos` : to store videos for practicing
- `runs` : to store the consequences









































