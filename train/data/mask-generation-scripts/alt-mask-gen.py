import os
import numpy as np

from skimage  import io
from scipy.io import loadmat
from PIL      import Image


def getData(source = "./ann_files"):
    images = dict()
    for r, d, f in os.walk(source):
        for file in f:
            if '.ann' in file and '.ann' == file[-4:]:
                fileContents = loadmat(os.path.join(r, file))['persistentData']
                for sectionIndex in range(len(fileContents)):
                    if(sectionIndex%2==0):
                        for data in fileContents[sectionIndex:sectionIndex+1]:
                            if(len(data[0]) > 0):
                                imageName = data[0][0]
                                stackHeight = io.imread("./images/" + imageName[:6] + "/"+ imageName[:-4] + ".tif").shape[0]
                                images[imageName] = dict()
                                images[imageName]['dendrite'] = [data[1][0][0][1], data[1][0][0][2]] #[polyX, polyY]
                                images[imageName]['spines'] = []
                                for i in range(3, len(data[1][0])):
                                    z = data[1][0][i][3][0][0] - 1#layer height in stack is one-indexed in ann files
                                    
                                    x0 = data[1][0][i][1][0][0]
                                    x1 = data[1][0][i][1][0][0]
                                    y0 = data[1][0][i][2][0][0]
                                    y1 = data[1][0][i][2][1][0]

                                    images[imageName]['spines'].append([x0, x1, y0, y1, z])

    return images
                                
                                
def generateSpineMasks(destination = "./trainingData/", numDims = 3):
    count = 0
    dataDict = getData()
    for imageName in dataDict:
        imgStack    = io.imread("./images/" + imageName[:6] + "/"+ imageName[:-4] + ".tif")
        stackHeight = imgStack.shape[0]
        layerMasks  = np.zeros((stackHeight, 512, 512), dtype = int) #initialize the masks to 3d array of 0's
        for spine in dataDict[imageName]['spines']:
            x0 = spine[0]
            x1 = spine[1]
            y0 = spine[2]
            y1 = spine[3]
            z  = spine[4]
            
            deltaX = x1-x0
            deltaY = y1-y0
            
            centerX = int(x0 + (deltaX)/2)
            centerY = int(y0 + (deltaY)/2)
            
            radius  = int(((deltaX**2 + deltaY**2)**0.5) / 2)
            
            for x in range(centerX - radius, centerX + radius):
                for y in range(centerY - radius, centerY + radius):
                    if (centerX - x)**2 + (centerY - y)**2 < radius ** 2:
                        if x in range(512) and y in range(512):
                            layerMasks[z][y][x] = 255
        
        if numDims == 3:
            #save the image and mask as tifs in the destination directory:
            io.imsave(destination + "images_3D/" + imageName[:-4] + ".tif", imgStack  )
            io.imsave(destination + "spineMasks_3D/"  + imageName[:-4] + ".tif", layerMasks)
            count += 1
        elif numDims == 2:
            for z in range(stackHeight):
                img  = Image.fromarray(imgStack[z,:,:].astype(np.uint8))
                mask = Image.fromarray(layerMasks[z,:,:].astype(np.uint8))
                img.save(destination + "images_2D/" + imageName[:-4] + "_" + str(z) + ".png")
                mask.save(destination + "spineMasks_2D/" + imageName[:-4] + "_" + str(z) + ".png")
                count += 1
        else:
            print("numDims must be 2 or 3")
            return
    print("Saved " + str(count) + " " + str(numDims) + "D spine masks")
        
"""
helper function for generateDenMasks
"""
def fillCircle(img, x, y, r):
    for i in range(max(0 ,int(x - r)), min(511, int(x + r))):
        for j in range(max(0, int(y - r)), min(511, int(y + r))):
            if (x - i) ** 2 + (y - j) ** 2 < r**2:
                img[j,i] = 255
    return img

"""
makes and saves dendrite masks which coorespond to max images
"""
def generateDenMasks(destination = "./trainingData/"):
    count = 0
    dataDict = getData()
    for imageName in dataDict:
##        denMask  = np.zeros((512, 512), dtype = int) #initialize the mask to 2d array of 0's
        denMask = io.imread("./trainingData/dendriteImages/" + imageName[:-4] + ".png")

        polyX = dataDict[imageName]['dendrite'][0]
        polyY = dataDict[imageName]['dendrite'][1]
        for i in range(len(polyX) - 1):
            x0 = int(polyX[i])
            y0 = int(polyY[i])
            x1 = int(polyX[i + 1])
            y1 = int(polyY[i + 1])
            dX = x1 - x0
            dY = y1 - y0
            if dX == 0:
                for y in range(min(y0, y1),max(y0,y1)):
                    denMask = fillCircle(denMask, x0, y, 30)             
            else:
                slope = dY/dX
                if abs(dX) >= abs(dY):   
                    for x in range(min(x0, x1), max(x0,x1)):
                        y = y0 + (x - x0) * slope
                        denMask = fillCircle(denMask, x, y, 30)
                else:
                    for y in range(min(y0, y1), max(y0, y1)):
                        x = ((y - y0) / slope) + x0
                        denMask = fillCircle(denMask, x, y, 30)

        mask = Image.fromarray(denMask.astype(np.uint8))
        mask.save(destination + "dendriteMasksWide/" + imageName[:-4] + ".png")
        count += 1
        
    print("Saved " + str(count) + " dendrite masks")

"""
finds all .max images and saves them in destination
"""
def collectMaxImages(source = "./images/", destination = "./trainingData"):
    count = 0
    for r, d, f in os.walk(source):
        for file in f:
            if '.tif' in file and '.tif' == file[-4:] and 'max' in file: 
                img = Image.open(os.path.join(r, file))
                img = Image.fromarray(np.array(img).astype(np.uint8))
                img.save(destination + "/maxImages/" + file[:10] + file[13:-4] + '.png')
                count += 1
            
    print("Saved " + str(count) + " max images")

"""
Finds and copies all of the max images which have a cooresponding dendrite mask
Necessary since not all images have labeled dendrites
"""
def selectMaxImages():
    count = 0
    masks = set()
    for r, d, f in os.walk("./trainingData/dendriteMasks"):
        for file in f:
            masks.add(file)
    for r, d, f in os.walk("./trainingData/maxImages"):
        for file in f:
            if file in masks:
                img = Image.open(os.path.join(r, file))
                img = Image.fromarray(np.array(img).astype(np.uint8))
                img.save("./trainingData/dendriteImages/" + file)
                count += 1
    print("Saved " + str(count) + " max images to pair with dendrite masks")




def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    from: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0
"""
generates a binary (0 or 255) image for each stack.
0 pixels are areas in the image that can be removed
from the original image since there may be unlabeled spines and dendentrites there
"""
def generateDendriteOutline(destination = "./trainingData/"):
    count = 0
    dataDict = getData()
    for imageName in dataDict:
        img    = np.zeros((512,512), dtype = int)#io.imread("./images/" + imageName[:6] + "/"+ imageName[:-4] + ".tif")
        
        polyX = dataDict[imageName]['dendrite'][0]
        polyY = dataDict[imageName]['dendrite'][1]
        hull = [(polyX[0], polyY[0]), (polyX[-1], polyY[-1])]
        for spine in dataDict[imageName]['spines']:
            x1 = spine[1]
            y1 = spine[3]
            hull.append((x1,y1))

            
        for x in range(512):
            for y in range(512):
                if in_hull((x,y), hull):
                    img[y,x] = 255 #TODO change to 0 once verified
                        
    img  = Image.fromarray(img.astype(np.uint8))
    img.save(destination + "images_cropped_2D/" + imageName[:-4] + ".png")
    count += 1
    print(count)  
       

##generateSpineMasks(numDims = 3)
##generateSpineMasks(numDims = 2)
generateDenMasks()
##collectMaxImages()
##selectMaxImages()

