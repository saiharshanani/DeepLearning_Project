Steps for Object Recognition:

1. Convert to JPEG and arrange the files in numeric order - convert_to_jpeg.py
2 . To label the images use main.py
3. Convert to YOLO format - convert_to_yolo.py
4. Get train.txt and test.txt - process.py where few images are seperated for training and testing
5. Train and test models on darknet using the dataset created 
    https://github.com/AlexeyAB/darknet
6. For Training the model execute the command - ./darknet detector train cfg/obj.data cfg/yolo-obj.cfg darknet19_448.conv.23.
7. For Testing the model execute the command - ./darknet detector test cfg/obj.data cfg/yolo-obj.cfg yolo-obj1000.weights data/testimage.jpg
