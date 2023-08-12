import cv2
import os
import ast
import numpy as np
from ultralytics import YOLO
import pandas as pd
import pickle
from tkinter import messagebox

#YOLO trained Model and Linear Regression Model 
model = YOLO('../Models/best.pt')
with open('../Models/timer_model.pkl', 'rb') as file:
    timer = pickle.load(file)

# Global Declarations 
area = []
count_lst = []
completed = False
time = int(str(np.random.randint(1, 20, 1)[0]) + '000')

# Mouse event callback function
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(area) <= 4:
            area.append((x, y))
        if(len(area) == 4):
            cv2.destroyAllWindows()
        elif(len(area) < 4):
            cv2.waitKey(0)
            
def capture():
    # Open the video file
    video_path = '..\\Data\\heavy_traffic.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get the desired frame time in milliseconds
    frame_time_ms = time  # Example: 5000 milliseconds (5 seconds)

    # Set the video capture position to the desired frame time
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time_ms)

    # Read the frame at the specified time
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if ret:
        return frame

def defining_area():
    # Load the image
    imig = capture()
    image = cv2.resize(imig, (900, 600))

    # Create a window and set the mouse callback
    cv2.namedWindow("Image - Select a Rectangular Area using Mouse Clicks")
    cv2.setMouseCallback("Image - Select a Rectangular Area using Mouse Clicks", get_coordinates)

    # Display the image
    cv2.imshow("Image - Select a Rectangular Area using Mouse Clicks", image)

    # Wait for the user to click on the image
    key = cv2.waitKey(0)
    if key == 27:
        messagebox.showerror("Alert", "Complete this step First")

    # Close all windows
    #cv2.destroyAllWindows()

    with open('..\\Data\\area.txt', 'w') as f:
        f.write(str(area))


def count_vehicles():
    with open('..\\Data\\area.txt', 'r') as f:
        res = ast.literal_eval(f.readline())
    
    my_file = open('..\\Data\\classes.txt', 'r')
    classes = my_file.read()
    class_lst = classes.split('\n')

    imig = capture()
    image = cv2.resize(imig, (900, 600))

    cv2.namedWindow("Image")

    frame = model.predict(image)
    results = frame[0].boxes.data
    pixels = pd.DataFrame(results).astype('float')

    # Define the points for the polygon
    point1 = res[0]  # Top-left corner
    point2 = res[1]  # Top-right corner
    point3 = res[2]  # Bottom-right corner
    point4 = res[3]  # Bottom-left corner
    points = np.array([point1, point2, point3, point4], np.int32)

    for index, row in pixels.iterrows():
        
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        d = int(row[5])
        c = class_lst[d]

        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2

        ret = cv2.pointPolygonTest(points, (cx, cy), False)

        if ret >= 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 3, (255, 0, 255), -1)
            cv2.putText(image, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

            count_lst.append(c)

    cv2.polylines(image, [points], True, (255, 255, 0), 2)
    
    vehicle_count = np.array(len(count_lst)).reshape(-1, 1)

    total_vehicles = str("Total Number of Vehicles : {}".format(len(count_lst)))
    time_allocated = str("Time Allocated : {0} seconds".format(int(timer.predict(vehicle_count)[0][0])))
    
    print("\n", total_vehicles, "\n", time_allocated, "\n")

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open('..\\Data\\area.txt', 'w') as file:
        pass

    
# Needs to Enter the Absolute Address that's why this type of Address
if os.path.getsize('..\\Data\\area.txt') == 0:
    while(len(area) < 4):
        defining_area()

count_vehicles()



   