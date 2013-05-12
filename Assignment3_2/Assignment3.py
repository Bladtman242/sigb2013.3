'''
Created on Apr 11, 2013

@author: Diako Mardanbegi (dima@itu.dk)
'''
import numpy as np
from pylab import *
from scipy import linalg
import cv2 as cv2
import cv2.cv as cv
from SIGBTools import *


global cam1, firstView

def getCornerCoords(boardImg):

    """
    :param boardImg:
    :return: cornercoords and centercoord
    """
    patternSize = (9,6)
    found,corners=cv2.findChessboardCorners(boardImg, patternSize)
    if(found):
        (x1,y1) = (corners[0][0][0],corners[0][0][1])
        (x2,y2) = (corners[7][0][0],corners[7][0][1])
        (x3,y3) = (corners[45][0][0],corners[45][0][1])
        (x4,y4) = (corners[53][0][0],corners[53][0][1])
        return True,[(x4,y4),(x3,y3),(x1,y1),(x2,y2)]
    else: return False, []

def DrawLines(img, points):
    for i in range(1, 17):                
         x1 = points[0, i - 1]
         y1 = points[1, i - 1]
         x2 = points[0, i]
         y2 = points[1, i]
         cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0),5) 
    return img


def getCubePoints(center, size,chessSquare_size):

    """ Creates a list of points for plotting
    a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """    
    points = []
    
    """
    1    2
        5    6
    3    4
        7    8 (bottom)
    
    """

    #bottom
    points.append([center[0]-size, center[1]-size, center[2]-2*size])#(0)5
    points.append([center[0]-size, center[1]+size, center[2]-2*size])#(1)7
    points.append([center[0]+size, center[1]+size, center[2]-2*size])#(2)8
    points.append([center[0]+size, center[1]-size, center[2]-2*size])#(3)6
    points.append([center[0]-size, center[1]-size, center[2]-2*size]) #same as first to close plot
    
    #top
    points.append([center[0]-size,center[1]-size,center[2]])#(5)1
    points.append([center[0]-size,center[1]+size,center[2]])#(6)3
    points.append([center[0]+size,center[1]+size,center[2]])#(7)4
    points.append([center[0]+size,center[1]-size,center[2]])#(8)2
    points.append([center[0]-size,center[1]-size,center[2]]) #same as first to close plot
    
    #vertical sides
    points.append([center[0]-size,center[1]-size,center[2]])
    points.append([center[0]-size,center[1]+size,center[2]])
    points.append([center[0]-size,center[1]+size,center[2]-2*size])
    points.append([center[0]+size,center[1]+size,center[2]-2*size])
    points.append([center[0]+size,center[1]+size,center[2]])
    points.append([center[0]+size,center[1]-size,center[2]])
    points.append([center[0]+size,center[1]-size,center[2]-2*size])
    points=dot(points,chessSquare_size)
    return array(points).T

def getCameraMethod1(currentFrame):
    globals()
    I1Gray = cv2.cvtColor(firstView, cv2.COLOR_RGB2GRAY)
    I2Gray = cv2.cvtColor(currentFrame, cv2.COLOR_RGB2GRAY)
    I1Corners = getCornerCoords(I1Gray)[1]
    I2Corners = getCornerCoords(I2Gray)[1]

    if(I1Corners == None or I2Corners == None):
        return
    H = estimateHomography(I1Corners, I2Corners)
    # compute second camera matrix from cam1.
    return Camera(dot(H, cam1.P))


def getCameraMethod2(currentFrame, distortCoefs):
    ##object points
    pattern_size=(9,6)
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= chessSquare_size
    obj_points = [pattern_points]
    obj_points.append(pattern_points)
    obj_points = np.array(obj_points,np.float64).T
    obj_points=obj_points[:,:,0].T
    K,r,t = cam1.factor()
    #--------
    fou, imPoints = cv2.findChessboardCorners(currentFrame,(9,6))
    if(not fou):
        return None
    try:
        found, rvecs_new, tvecs_new = GetObjectPos(obj_points,imPoints,K,distortCoefs)
    except ValueError:
        rvecs_new, tvecs_new = GetObjectPos(obj_points,imPoints,K,distortCoefs)
        found = True;

    #if(not found):
    #    return None
    rvecs_new = cv2.Rodrigues(np.array(rvecs_new))[0]
    return Camera(np.dot(K, np.hstack((rvecs_new, tvecs_new))))

def doCulling(camera, cent, norm):
    #camera center in world coords
    camera_center = array(camera.center()).T

    #vector from cam center to surface center.
    lookVector = camera_center - cent

    #normalize the vector
    lookVector = lookVector/np.linalg.norm(lookVector)

    angle = Angle3D(lookVector[0], norm)
    if (angle > 89):
        return False
    else: return True


def drawSurfaceVector(img, cent, norm, camera):
    normTip = cent + norm

    cent_proj = camera.project(toHomogenious(np.array([cent]).T))

    normTip_proj = camera.project(toHomogenious(np.array([normTip]).T))

    cv2.line(img, (cent_proj[0],cent_proj[1]), (normTip_proj[0],normTip_proj[1]), (0,255,255), 3)



    #p = camera.project((1,1))
    #p = (p[0],p[1])
    #p = np.dot(camera.t,p)

    #cv2.circle(img,((lookVector_proj[0]/lookVector_proj[2], lookVector_proj[1]/lookVector_proj[2])),10,(255,0,255))
    return img

def getSurfaceVectors(face, camera):

    a = np.array([face[0][0],face[1][0], face[2][0]]).T
    #a = np.array(np.dot(camera.P,a))[0]

    b = np.array([face[0][1],face[1][1], face[2][1]]).T
    #b = np.array(np.dot(camera.P,b))[0]

    c = np.array([face[0][2],face[1][2], face[2][2]]).T
    #c = np.array(np.dot(camera.P,c))[0]

    a = np.array([a[0], a[1],a[2]])
    b = np.array([b[0], b[1],b[2]])
    c = np.array([c[0], c[1],c[2]])

    #vector between two face points
    v_ba = np.array([a[0]-b[0],a[1]-b[1],a[2]-b[2]])
    v_bc = np.array([c[0]-b[0],c[1]-b[1],c[2]-b[2]])
    #surface center
    cent = ((v_ba+v_bc)/2) + b

    #normalize norm --- ghurt is working here.
    norm = np.cross(v_ba,v_bc)/np.linalg.norm(np.cross(v_ba,v_bc))

    #surface center and norm
    return (cent, norm)

def addTexWeighted(img, tex, Face, camera):
    ITop = tex
    mTop,nTop,i = shape(ITop)
    topTexPoints = [(0,0),(0,nTop),(mTop,nTop),(mTop,0)]
    side_cam = np.array(camera.project(toHomogenious(Face)))

    #toppen
    sideCam = [(int(side_cam[0][0]),int(side_cam[1][0])), (int(side_cam[0][1]),int(side_cam[1][1])),
              (int(side_cam[0][2]),int(side_cam[1][2])), (int(side_cam[0][3]),int(side_cam[1][3]))]
    #making a H between texture and side of cube
    H = estimateHomography(topTexPoints, sideCam)
    m,n,d = shape(img)
    overlay = cv2.warpPerspective(ITop,H,(n,m))
    img = cv2.addWeighted(img,0.5,overlay,0.5,0)
    return img

def addTexMask(img, tex, Face, camera):
    ITop = tex
    white = np.copy(ITop)

    white[:,:] = [255,255,255]
    mTop,nTop,i = shape(ITop)
    topTexPoints = [(0,0),(0,nTop),(mTop,nTop),(mTop,0)]
    side_cam = np.array(camera.project(toHomogenious(Face)))

    #toppen
    sideCam = [(int(side_cam[0][0]),int(side_cam[1][0])), (int(side_cam[0][1]),int(side_cam[1][1])),
              (int(side_cam[0][2]),int(side_cam[1][2])), (int(side_cam[0][3]),int(side_cam[1][3]))]
    #making a H between texture and side of cube
    H = estimateHomography(topTexPoints, sideCam)
    m,n,d = shape(img)
    overlay = cv2.warpPerspective(white,H,(n,m))
    texWarped = cv2.warpPerspective(ITop,H,(n,m))
    Mask = cv2.threshold(overlay,1,255,cv2.THRESH_BINARY)[1]
    backGr = 255 - Mask
    I1 = cv2.bitwise_and(backGr, img)
    img = cv2.bitwise_or(I1, texWarped)

    return img

def update(img):
    globals()
    Undistorting = True;
    if Undistorting:  #Use previous stored camera matrix and distortion coefficient to undistort the image
        distortC = np.load("data/numpyData/distortionCoefficient.npy")
        image = cv2.undistort(img, cameraMat, distortC)
    
    if (ProcessFrame):
        I1grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        found, I1corners = getCornerCoords(I1grey)
        patternFound=found
    
        if patternFound ==True:        
            
            ''' <006> Here Define the cameraMatrix P=K[R|t] of the current frame'''

            if(debug):
                #P2_METHOD2
                camera = getCameraMethod2(img, distortC)
                if camera == None:
                    return
                camera.factor()
            else:
                #P2_METHOD1
                camera = getCameraMethod1(img)
                if camera == None:
                    return
                camera.factor()




            if ShowText:
                ''' <011> Here show the distance between the camera origin and the world origin in the image'''

                K,R,T = camera.factor()
                x,y,z = T
                dist = sqrt(pow(x,2)+pow(y,2)+pow(z,2))
                cv2.putText(img,str("frame:" + str(frameNumber) +str(" Dist: ")+ str(dist)), (20,10),cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 255))#Draw the text

                ''' <008> Here Draw the world coordinate system in the image'''
                #okay kids look below.. four point. first is origin, next three are x,y,z
                o = np.matrix([0,0,0,1])
                x = np.matrix([3,0,0,1])
                y = np.matrix([0,3,0,1])
                z = np.matrix([0,0,-3,1])

                originCoord = [o.T,x.T,y.T,z.T]
                for point in originCoord:
                    origin = camera.project(originCoord[0])
                    poin = camera.project(point)
                    cv2.line(img,(int(origin[0]),int(origin[1])),(int(poin[0]),int(poin[1])),(255,255,0),2)
            
            if TextureMap:


                ''' <012> Here draw the surface vectors'''

                cent,topNorm = getSurfaceVectors(TopFace,camera)
                cent,leftNorm = getSurfaceVectors(LeftFace,camera)
                cent,rightNorm = getSurfaceVectors(RightFace,camera)
                cent,downNorm = getSurfaceVectors(DownFace,camera)
                cent,upNorm = getSurfaceVectors(UpFace,camera)

                drawSurfaceVector(img, cent, topNorm, camera)
                ''' <013> Here Remove the hidden faces'''
                drawTop = doCulling(camera, cent, topNorm)
                drawLeft = doCulling(camera, cent, leftNorm)
                drawRight = doCulling(camera, cent, rightNorm)
                drawDown = doCulling(camera, cent, downNorm)
                drawUp = doCulling(camera, cent, upNorm)

                TopFaceCornerNormals,RightFaceCornerNormals,LeftFaceCornerNormals,UpFaceCornerNormals,DownFaceCornerNormals =CalculateFaceCornerNormals(TopFace,RightFace,LeftFace,UpFace,DownFace)

                ''' <010> Here Do the texture mapping and draw the texture on the faces of the cube'''
                if(drawTop):
                    ITop = cv2.imread("data/Images3/Top.jpg")
                    img = addTexMask(img,ITop,TopFace, camera)
                if(drawLeft):
                    ILeft = cv2.imread("data/Images3/Left.jpg")
                    img = addTexMask(img,ILeft,LeftFace, camera)
                if(drawRight):
                    IRight = cv2.imread("data/Images3/Right.jpg")
                    img = addTexMask(img,IRight,RightFace, camera)
                if(drawDown):
                    IDown = cv2.imread("data/Images3/Down.jpg")
                    img = addTexMask(img,IDown,DownFace, camera)
                if(drawUp):
                    IUp = cv2.imread("data/Images3/Up.jpg")
                    img = addTexMask(img,IUp,UpFace, camera)
                # img = addTexWeighted(img,ITop,TopFace, camera)
                # img = addTexWeighted(img,ILeft,LeftFace, camera)
                # img = addTexWeighted(img,IRight,RightFace, camera)
                # img = addTexWeighted(img,IDown,DownFace, camera)
                # img = addTexWeighted(img,IUp,UpFace, camera)

                img=ShadeFace(img,TopFace,TopFaceCornerNormals,camera)
                img=ShadeFace(img,RightFace,RightFaceCornerNormals,camera)
                img=ShadeFace(img,LeftFace,LeftFaceCornerNormals,camera)
                img=ShadeFace(img,UpFace,UpFaceCornerNormals,camera)
                img=ShadeFace(img,DownFace,DownFaceCornerNormals,camera)

            if ProjectPattern:
                ''' <007> Here Test the camera matrix of the current view by projecting the pattern points'''
                points = toHomogenious(np.array(np.load("data/numpyData/obj_points.npy"))[0].T)
                projected_points = camera.project(points)
                for point in projected_points.T:
                    cv2.circle(img,(int(point[0,0]),int(point[0,1])),5,(255,255,0),2)


            if WireFrame:                      
                ''' <009> Here Project the box into the current camera image and draw the box edges'''
                # box = getCubePoints((0,0,0),2,2)
                box_cam = camera.project(toHomogenious(box))
                DrawLines(img,box_cam)
    cv2.imshow('Web cam', img)
    global result
    result=copy(img)


def Angle3D(v1,v2):
    #vectot lengths
    # l1=np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    # l2=np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    ca = np.dot(v1,v2)
    return acos(ca)*180/math.pi

def RobustAngle3D(v1,v2):
    #vectot lengths
    l1=math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    l2=math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    ca = np.dot(v1,v2) / (l1 * l2)
    return acos(ca)*180/math.pi

def getImageSequence(capture, fastForward):
    '''Load the video sequence (fileName) and proceeds, fastForward number of frames.'''
    global frameNumber
   
    for t in range(fastForward):
        isSequenceOK, originalImage = capture.read()  # Get the first frames
        frameNumber = frameNumber+1
    return originalImage, isSequenceOK


def printUsage():
    print "Q or ESC: Stop"
    print "SPACE: Pause"     
    print "p: turning the processing on/off "  
    print 'u: undistorting the image'
    print 'i: show info'
    print 't: texture map'
    print 'g: project the pattern using the camera matrix (test)'
    print 's: save frame'
    print 'x: do something!'

def ShadeFace(image,points,faceCorner_Normals, camera):

    global shadeRes

    shadeRes=10

    videoHeight, videoWidth, vd = array(image).shape

#................................

    points_Proj=camera.project(toHomogenious(points))

    points_Proj1 = np.array([[int(points_Proj[0,0]),int(points_Proj[1,0])],[int(

    points_Proj[0,1]),int(points_Proj[1,1])],[int(points_Proj[0,2]),int(points_Proj

    [1,2])],[int(points_Proj[0,3]),int(points_Proj[1,3])]])

#................................

    square = np.array([[0, 0], [shadeRes-1, 0], [shadeRes-1, shadeRes-1], [0, shadeRes-

    1]])

#................................

    H = estimateHomography(square, points_Proj1)

#................................

    Mr0,Mg0,Mb0=CalculateShadeMatrix(image,shadeRes,points,faceCorner_Normals, camera,255)

# HINT

# type(Mr0): <type 'numpy.ndarray'>

# Mr0.shape: (shadeRes, shadeRes)

#................................

    Mr = cv2.warpPerspective(Mr0, H, (videoWidth, videoHeight),flags=cv2.INTER_LINEAR)

    Mg = cv2.warpPerspective(Mg0, H, (videoWidth, videoHeight),flags=cv2.INTER_LINEAR)

    Mb = cv2.warpPerspective(Mb0, H, (videoWidth, videoHeight),flags=cv2.INTER_LINEAR)

#................................

    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    [r,g,b]=cv2.split(image)

#................................

    whiteMask = np.copy(r)

    whiteMask[:,:]=[0]

    points_Proj2=[]

    points_Proj2.append([int(points_Proj[0,0]),int(points_Proj[1,0])])

    points_Proj2.append([int(points_Proj[0,1]),int(points_Proj[1,1])])

    points_Proj2.append([int(points_Proj[0,2]),int(points_Proj[1,2])])

    points_Proj2.append([int(points_Proj[0,3]),int(points_Proj[1,3])])

    cv2.fillConvexPoly(whiteMask,np.array(points_Proj2).astype('int32'),(255,255,255))

#................................

    r[nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),r[nonzero(whiteMask>0)]*Mr[

    nonzero(whiteMask>0)])

    g[nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),g[nonzero(whiteMask>0)]*Mg[

    nonzero(whiteMask>0)])

    b[nonzero(whiteMask>0)]=map(lambda x: max(min(x,255),0),b[nonzero(whiteMask>0)]*Mb[

    nonzero(whiteMask>0)])

#................................

    image=cv2.merge((r,g,b))

    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def fallOut(x):
    # return 1/(0.03 * x**2 + 0*x + 0)
    return 1

def vecLen3D(x):
    return math.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

#l is the direction of the light source 
#n is the normal to the surface at the point p
# Where is ID? Where is IS?
def CalculatePhongIlluminationModel(n,l,r,v,IA,IS,ID,KA,KS,KD):
    #(IA*KA)+(ID*KD*max(n*l,0))
    pass;

def CalculateDiffuse(lightSrc, endPoint, faceCorner_Normals, kd,IL):

    lightVector = np.array([
            lightSrc[0]-endPoint[0],
            lightSrc[1]-endPoint[1],
            lightSrc[2]-endPoint[2]
            ])

    l = lightVector / vecLen3D(lightVector)
    l = l.T[0]
    print "l",type(l), shape(l), l 
    
    n = faceCorner_Normals.T[0] 
    n = n / vecLen3D(n)
    
    print "n",type(n), shape(n), n 

    print "dot", np.dot(n,l)
    dot = max((np.dot(n, l)),0)

    resultR = np.array((IL[0] * kd[0]) * dot)[0][0]
    resultG = np.array((IL[1] * kd[1]) * dot)[0][0]
    resultB = np.array((IL[2] * kd[2]) * dot)[0][0]

    return (resultR,resultG,resultB)



def CalculateShadeMatrix(image,shadeRes,points,faceCorner_Normals,camera,intensity): 
    
    """
    Given in the assignment
    """
    #Ambient light IA=[IaR,IaG,IaB]

    IA = np.matrix([5.0, 5.0, 5.0]).T

    #Point light IA=[IpR,IpG,IpB]

    IP = np.matrix([5.0, 5.0, 5.0]).T
    
    # This is guesswork
    #ID = np.matrix([5.0, 5.0, 5.0]).T

    #Light Source Attenuation

    fatt = 1

    #Material properties: e.g., Ka=[kaR; kaG; kaB]

    ka=np.matrix([0.2, 0.2, 0.2]).T

    kd= np.matrix([0.3, 0.3, 0.3]).T

    ks=np.matrix([0.7, 0.7, 0.7]).T

    alpha = 100

    """
    End - Given in the assignment
    """

    #lightSrc = np.array(np.matrix([30,30,30])).T
    lightSrc = np.array(camera.center())
    
    endPoint = np.array([
            (points[0][0]+points[0][1]+points[0][2]+points[0][3])/4,
            (points[1][0]+points[1][1]+points[1][2]+points[1][3])/4,
            (points[2][0]+points[2][1]+points[2][2]+points[2][3])/4
            ])
    
    
    (Ir,Ig,Ib) = CalculateDiffuse(lightSrc, endPoint,faceCorner_Normals, kd, IP)
    
    arrR = np.ones((shadeRes,shadeRes))
    arrG = np.ones((shadeRes,shadeRes))
    arrB = np.ones((shadeRes,shadeRes))

    arrR[:] = Ir
    arrG[:] = Ig
    arrB[:] = Ib
    
    return (arrR, arrG, arrB)

def run(speed): 
    
    '''MAIN Method to load the image sequence and handle user inputs'''   

    #--------------------------------video
    capture = cv2.VideoCapture("Pattern6.mp4")
    #--------------------------------camera
    #capture = cv2.VideoCapture(0)

    image, isSequenceOK = getImageSequence(capture,speed)       

    if(isSequenceOK):
        update(image)
        printUsage()

    while(isSequenceOK):
        OriginalImage=copy(image)
         
        
        inputKey = cv2.waitKey(1)
        
        if inputKey > 1000: # Linux hack
            inputKey =  inputKey % 256;

        if inputKey == 32:#  stop by SPACE key
            update(OriginalImage)
            if speed==0:     
                speed = tempSpeed;
            else:
                tempSpeed=speed
                speed = 0;                    
            
        if (inputKey == 27) or (inputKey == ord('q')):#  break by ECS key
            break    
                
        if inputKey == ord('p') or inputKey == ord('P'):
            global ProcessFrame
            if ProcessFrame:     
                ProcessFrame = False;
                
            else:
                ProcessFrame = True;
            update(OriginalImage)
            
        if inputKey == ord('u') or inputKey == ord('U'):
            global Undistorting
            if Undistorting:     
                Undistorting = False;
            else:
                Undistorting = True;
            update(OriginalImage)     
        if inputKey == ord('w') or inputKey == ord('W'):
            global WireFrame
            if WireFrame:     
                WireFrame = False;
                
            else:
                WireFrame = True;
            update(OriginalImage)

        if inputKey == ord('i') or inputKey == ord('I'):
            global ShowText
            if ShowText:     
                ShowText = False;
                
            else:
                ShowText = True;
            update(OriginalImage)
            
        if inputKey == ord('t') or inputKey == ord('T'):
            global TextureMap
            if TextureMap:     
                TextureMap = False;
                
            else:
                TextureMap = True;
            update(OriginalImage)
            
        if inputKey == ord('g') or inputKey == ord('G'):
            global ProjectPattern
            if ProjectPattern:     
                ProjectPattern = False;
                
            else:
                ProjectPattern = True;
            update(OriginalImage)   
             
        if inputKey == ord('x') or inputKey == ord('X'):
            global debug
            if debug:     
                debug = False;                
            else:
                debug = True;
            update(OriginalImage)   
            
                
        if inputKey == ord('s') or inputKey == ord('S'):
            name='Saved Images/Frame_' + str(frameNumber)+'.png' 
            cv2.imwrite(name,result)
           
        if (speed>0):
            update(image)
            image, isSequenceOK = getImageSequence(capture,speed)          



#---Global variables
global cameraMatrix    
global distortionCoefficient
global homographyPoints
global calibrationPoints
global calibrationCamera
global chessSquare_size
    
ProcessFrame=False
Undistorting=False   
WireFrame=False
ShowText=True
TextureMap=True
ProjectPattern=False
debug=True

frameNumber=0


chessSquare_size=2
            
box = getCubePoints([4, 2.5, 0], 1,chessSquare_size)            


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [0,3,2,1],[0,3,2,1] ,[0,3,2,1]  ])  # indices for the second dim            
TopFace = box[i,j]

i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [3,8,7,2],[3,8,7,2] ,[3,8,7,2]  ])  # indices for the second dim            
RightFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [5,0,1,6],[5,0,1,6] ,[5,0,1,6]  ])  # indices for the second dim            
LeftFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [5,8,3,0], [5,8,3,0] , [5,8,3,0] ])  # indices for the second dim            
UpFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim 
j = array([ [1,2,7,6], [1,2,7,6], [1,2,7,6] ])  # indices for the second dim            
DownFace = box[i,j]



'''----------------------------------------'''
'''----------------------------------------'''



''' <000> Here Call the cameraCalibrate2 from the SIGBTools to calibrate the camera and saving the data''' 
# calibrateCamera()
# cameraCalibrate2(5,(9,6),2.0,0)
# RecordVideoFromCamera()

''' <001> Here Load the numpy data files saved by the cameraCalibrate2''' 
cameraMat  = np.load("data/numpyData/camera_matrix.npy")
roVectors = np.load("data/numpyData/rotatioVectors.npy")
transVectors  = np.load("data/numpyData/translationVectors.npy")

''' <002> Here Define the camera matrix of the first view image (01.png) recorded by the cameraCalibrate2''' 

roVectors = cv2.Rodrigues(np.array(roVectors)[0])[0]
transVectors = np.array(transVectors)[0]

# print "rotationvector",roVectors
# print "transvector",transVectors


cam1 = Camera(np.dot(cameraMat, np.hstack((roVectors, transVectors))))

cam1.factor()

# box_cam1 = cam1.project(toHomogenious(box))


''' <003> Here Load the first view image (01.png) and find the chess pattern and store the 4 corners of the pattern needed for homography estimation''' 
firstView = cv2.imread("01.png")

# points = toHomogenious(np.array(np.load("data/numpyData/obj_points.npy"))[0].T)
# projected_points = cam1.project(points)

# for point in projected_points.T:
#    cv2.circle(firstView,(int(point[0,0]),int(point[0,1])),5,(255,255,0),2)

# cv2.imshow("myss",firstView)
# cv2.waitKey(0)



run(1)
# vim: set ts=4:shiftwidth=4:expandtab:
