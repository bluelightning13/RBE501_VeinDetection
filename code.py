import cv2
import numpy
from random import randint
from matplotlib import pyplot as plt
from matplotlib import pylab as cm
import RPi.GPIO as GPIO
import os
import glob

def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(7, GPIO.OUT)
    GPIO.setup(11, GPIO.OUT)
    GPIO.setup(12, GPIO.OUT)

    GPIO.setup(15, GPIO.OUT)
    GPIO.setup(16, GPIO.OUT)
    GPIO.setup(21, GPIO.OUT)
    GPIO.setup(22, GPIO.OUT)

    GPIO.output(11, 1)
    GPIO.output(12, 1)
    GPIO.output(15, 1)
    GPIO.output(16, 1)
    GPIO.output(21, 1)
    GPIO.output(22, 1)

def calibration():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:5].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints1 = [] # 2d points in image plane.
    imgpoints2 = [] # 2d points in image plane.
    images1 = glob.glob('c1/*jpg')
    images2 = glob.glob('c2/*jpg')

    lenImg = len(images1)

    for fname in range(0,lenImg):
        #reads both camera images and resizes them 
        #(known bug with findCheckerboard with images to big)
        img1 = cv2.imread(images1[fname])
        img1 = cv2.resize(img1,(0,0),fx=.25, fy=.25)
        img2 = cv2.imread(images2[fname])
        img2 = cv2.resize(img2,(0,0),fx=.25, fy=.25)

        print"Loading images..."
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret1, corners1 = cv2.findChessboardCorners(gray1, (5,6))#,flags = cv2.CALIB_CB_FAST_CHECK)
        ret2, corners2 = cv2.findChessboardCorners(gray2, (5,6))
        
        # If found in both - add object points and image points (after refining them)
        if ret1 and ret2:
            objpoints.append(objp)
            corners1 = cv2.cornerSubPix(gray1,corners1,(11,11),(-1,-1),criteria)
            imgpoints1.append(corners1)
            corners2 = cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
            imgpoints2.append(corners2)

            #Draw and display the corners
            #img = cv2.drawChessboardCorners(img, (5,6), corners,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey()

    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    imgsize = gray.shape
    #ensure that at least three points are found - this can be increased for
    #more accurate calibration
    if(len(objpoints)>3):
        print "Images calibrated successfully"
        #print imgsize
        #cv2.destroyAllWindows()
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, imgsize, None, None)
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, imgsize, None, None)
        return r1,r2,p1,p2,q
    else:
        print "Please calibrate again!"




def filterImg(r1,r2,p1,p2,q):
    imgL = cv2.imread('c1/capture_1.jpg',0)
    imgR = cv2.imread('c2/capture_1.jpg',0)

    stereo = cv2.createStereoBM(numDisparities=16,blockSize=15)
    disparity = stereo.compute(imgL, imgR)

    #Median Blur
    print "Bluring..."
    imgL = cv2.medianBlur(imgL,9)
    imgR = cv2.medianBlur(imgR,9)

    filterImg = cv2.imread('filter2.png', 0)
    imgL = cv2.resize(imgL, (0,0), fx = .79, fy = .79)
    imgR = cv2.resize(imgR, (0,0), fx = .79, fy = .79)
    filterImg = cv2.resize(filterImg,(0,0), fx = 1, fy=.75)

    print "Starting dft..."

    dft_shiftL = numpy.fft.fftshift(cv2.dft(numpy.float32(imgL), flags = cv2.DFT_COMPLEX_OUTPUT))
    dft_shiftR = numpy.fft.fftshift(cv2.dft(numpy.float32(imgR), flags = cv2.DFT_COMPLEX_OUTPUT))

    rows, cols = imgL.shape
    crow, ccol = rows/2, cols/2
    mask = numpy.zeros((rows, cols,2), numpy.uint8)
    #filterImg = filterImg[484:1564,64:1984]
    mask[:,:,0] = filterImg[:,:]
    mask[:,:,1] = filterImg[:,:]


    img_backL = cv2.idft(numpy.fft.ifftshift(dft_shiftL*mask))
    img_backR = cv2.idft(numpy.fft.ifftshift(dft_shiftR*mask))

    img_backR = cv2.magnitude(img_backR[:,:,0], img_backR[:,:,1])
    img_backR = cv2.normalize(img_backR, 0, 255, cv2.NORM_MINMAX)
    img_backL = cv2.magnitude(img_backL[:,:,0], img_backL[:,:,1])
    img_backL = cv2.normalize(img_backL, 0, 255, cv2.NORM_MINMAX)

    print "Thresholding..."
    retR,img_threshR = cv2.threshold(img_backR,.135,255,cv2.THRESH_BINARY)
    retL,img_threshL = cv2.threshold(img_backL,.135,255,cv2.THRESH_BINARY)
    #cv2.namedWindow('thresholding', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('thresholding',cols/2, rows/2)
    #cv2.imshow("thresholding", img_thresh)


    print "Erosion and Dilation..."
    eltC = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    eltR = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    
    eroR = cv2.dilate(img_threshR,eltC, iterations=10)
    diaR = cv2.erode(eroR,eltR, iterations=30)
    diaR = cv2.dilate(diaR,eltR,iterations=15)

    eroL = cv2.dilate(img_threshL,eltC, iterations=10)
    diaL = cv2.erode(ero,eltRL, iterations=30)
    diaL = cv2.dilate(dia,eltRL,iterations=15)

    #cv2.namedWindow('DilationErosion', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('DilationErosion',cols/2, rows/2)
    #cv2.imshow("DilationErosion", dia)
    mask = numpy.zeros((rows+2,cols+2), numpy.uint8)
    #flood fill the corners
    print "Floodfill..."
    cv2.floodFill(diaR, mask, (0,0), 255)
    cv2.floodFill(diaR, mask, (2047, 1535), 255)
    cv2.floodFill(diaR, mask, (0, 1535), 255)
    cv2.floodFill(diaR, mask, (2047, 0), 255)

    cv2.floodFill(diaL, mask, (0,0), 255)
    cv2.floodFill(diaL, mask, (2047, 1535), 255)
    cv2.floodFill(diaL, mask, (0, 1535), 255)
    cv2.floodFill(diaL, mask, (2047, 0), 255)
    print "starting line detection..."
    #crop and do region
    cpR = 255 - diaR
    cpL = 255 - diaL
    

    size = numpy.size(cpL)
    skelR = numpy.zeros(cpR.shape,numpy.float32) #uint8
    skelL = numpy.zeros(cpL.shape,numpy.float32) #uint8
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    finished = False

    print "Starting skeletonization"
    while(not finished):
        eroded = cv2.erode(cpL,kernel)
        temp = cv2.dilate(eroded,kernel)
        temp = cv2.subtract(cpL,temp)
        skel = cv2.bitwise_or(skel,temp)
        cpL = eroded.copy()

        #zeros = size - cv2.countNonZero(img)
        #print zeros
        print cv2.countNonZero(cpL)
        if cv2.countNonZero(cpL)<10:
            finished = True
    print "Skeletonization 50%..."
    finished = False
    while(not finished):
        eroded = cv2.erode(cpR,kernel)
        temp = cv2.dilate(eroded,kernel)
        temp = cv2.subtract(cpR,temp)
        skel = cv2.bitwise_or(skel,temp)
        cpR = eroded.copy()

        #zeros = size - cv2.countNonZero(img)
        #print zeros
        print cv2.countNonZero(cpR)
        if cv2.countNonZero(cpR)<10:
            finished = True
            
    #veinL = skelL
    #veinR = skelR
    #vein = vein.astype(numpy.uint8)
    #over = cv2.addWeighted(img, .5, vein, .5, 0.0)
    #cv2.namedWindow('Vein', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Vein',cols/2, rows/2)
    #cv2.imshow("Vein", vein)
    #cv2.waitKey()
    
                
    #if the x are out of tolerance, highlight next section to indicate curve
    #points = cv2.findNonZero(vein)
    #points = cv2.findNonZero(vein)
    #change skel to rgb
    #finds the longest line that should be the vein and eliminates the other noise
    #may not be needed
    imgPtsL = longestLine(imgL, rows, cols)
    imgPtsR = longestLine(imgR, rows, cols)
    cXL = 0 
    cYL = 0
    cXR = 0
    cYR = 0
    numOfPts = len(imgPtsL)
    for img in range(0, numOfPts):
        cXL = cXL + (imgPtsL[img])[0] 
        cYL = cYL + (imgPtsL[img])[1]
        cXR = cXR + (imgPtsR[img])[0]
        cYR = cYR + (imgPtsR[img])[1]
    cXL = cXL/numOfPts
    cYL = cYL/numOfPts
    cXR = cXR/numOfPts
    cYR = cYR/numOfPts
    
    #this code is for displaying the vein line on the images. 
    #veinImage(skel) #if set to true, will return image that can be displayed
    
    print "Starting 3D Point Reconstruction..."
    #Finding real world point with disparity map (trajectory is in mm)
    stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 15)
    disparity = stereo.compute(img1, img2)
    difference = disparity(x,y)
    vectorPts = [cXL,cYL,difference(cXL,cYL),1]
    realWorldPts = Q*vectorPts
    realWorldPts_Normalized = [realWorldPts[0]/realWorldPts[3] , realWorldPts[1]/realWorldPts[3], realWorldPts[2]/realWorldPts[3]]
    print "Real world point in mm:"
    print realWorldPts_Normalized

    #points = Q*[x,y,disparity(x,y),1] #might need to tranpose Q
    #finalPoint = [points[0]/points[3],points[1]/points[3],points[2]/points[3]]
    return realWorldPts_Normalized

def veinImage(skel):
    colorImg = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    randClr = [0,0,0]
    rangepts = points.shape[0]
    a = points[0,0][0]
    clr = 0

    for i in range(0,rangepts):
        xpt = points[i,0][0]
        ypt = points[i,0][1]
        if (xpt > 1+a or xpt < a-1):
            #change color
            print "change"
            randClr = [clr+10, 0, 200]
            a = points[i,0][0]
        colorImg[ypt,xpt] = randClr

    ori = cv2.imread('capture_1.jpg')

    colorImg = colorImg.astype(numpy.uint8)
    over = cv2.addWeighted(ori, .5, colorImg, .5, 0.0)
    
    return over

def longestLine(img, rows, cols):
    notDone = False
    search = []
    visited = []
    oldVisited = []
    for u in range(0,cols):
        for u in range(0,rows):
            if (img[u,v] == 0):
                notDone = True
                img[u,v] == 255
                search.append[[u,v]]
            while(notDone):
                if (len(search)==0):
                    notDone == False
                search.pop(0)
                visited.append([u,v])
                for w in range(-1,1):
                    for z in range(-1,1):
                        if((img[u,v]|img[w,z])==1):
                            search.append([w,x])
                            img[w,z] = 255
            if(len(oldVisited) < len(visited)): #might need a null check here
                oldVisited = visited
    return oldVisited

def photoCapture(a):
    GPIO.output(7,0)
    GPIO.output(11,0)
    GPIO.output(12,1)
    capture(1, a)

    GPIO.output(7,0)
    GPIO.output(11,1) #here
    GPIO.output(12,0)
    capture(2, a)


def capture(folder, cam):
    cmd = "raspistill -t 500 -o c_%d/capture_%d.jpg" % folder, cam
    os.system(cmd)

def photoCaptureClibration():
    photoCapture(5)
    print "Press enter to continue to next image (1/5)"
    cv.waitKey()
    photoCapture(6)
    print "Press enter to continue to next image (2/5)"
    cv.waitKey()
    photoCapture(7)
    print "Press enter to continue to next image (3/5)"
    cv.waitKey()
    photoCapture(8)
    print "Press enter to continue to next image (4/5)"
    cv.waitKey()
    photoCapture(9)
    print "Press enter to continue to next image (5/5)"
    cv.waitKey()



if __name__ == "__main__":
    setup()
    #capture an image
    print "Calibration starting. Please have caibration image ready..."
    photoCaptureClibration()
    #calibrate device
    print "Computing Calibration..."
    r1,r2,p1,p2,q = calibration()
    print "Waiting for users. Press a key to continue..."
    cv2.waitKey()
    print "Starting Photo Capture"
    photoCapture(1)
    print "Starting processing..."
    finalpoints = filterImg(r1,r2,p1,p2,q):
    #Send Real world points 
    GPIO.output(7,0)
    GPIO.output(11,0)
    GPIO.output(12,1)
