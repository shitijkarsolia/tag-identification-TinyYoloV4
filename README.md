# Tag Identification Using Tiny-YoloV4
## Steps:
#### 1) Clone this repo:
```
git clone https://github.com/shitijkarsolia/tag-identification-TinyYoloV4.git
```

#### 2) Create and activate your virtualenv:
```
$ pip install virtualenv
$ virtualenv -p /usr/bin/python3 venv
$ source venv/bin/activate
```
#### 3) Install required packages:
```
pip install -r requirements.txt
```
#### 4) For single image inference:
Set the 'image_path' in line 55 single_image_yolo.py
```
$ python single_image_yolo.py
```
#### 4) For inference on folder with multiple images:
Set the 'image_dir' path in line 54 in multiple_image_yolo.py
```
$ python multiple_image_yolo.py
```

