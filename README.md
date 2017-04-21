# RBE501_VeinDetection WPI 4/2017
This code is for the vision part of a phlebotomy robot. It uses stereo vision to detect the vein's location. This was continuing a project that been done, however not much had been previously done on the vision system.

Libraries that are needed:
Numpy 1.12
OpenCV 3.1

Raspberry Pi Camera are used as that was the avaiable hardware. To get closer to real-time, new hardware is needed for the project. However the software has been optimised. 

1. First the camera's are calibrated using the calibration and photoCapture functions in code.py.
2. After they are calibrated, the user can insert their arm into the device. The code will wait for a user to press a key before starting.
3. Then by running the filering function, the images of the arm are taken, filtered and the vein points are determined. These points can then be used for the kinematic control of the needle.

The real world points that are returned are in mm. The reference frame is one of the cameras (it's a vector from the reference camera to the point) and will need to be put into world reference frame. This code should use the left camera as the reference, but it doesn't matter if switched to the right.


