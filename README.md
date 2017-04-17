# RBE501_VeinDetection WPI 4/2017
This code is for the vision part of a phlebotomy robot. It uses stereo vision to detect the vein's location. This was continuing a project that been done, however not much had been previously done on the vision system.

Libraries that are needed:
Numpy 1.12
OpenCV 3.1

Raspberry Pi Camera are used as that was the avaiable hardware. To get closer to real-time, new hardware is needed for the project. However the software has been optimised. 

1. First the camera's are calibrated using the calibration.py file.
2. After they are calibrated, the user can insert their arm into the device. 
3. Then by running the processing.py file, the images of the arm are taken, filtered and the vein points are determined. The raspberry pi sends the data over serial.


