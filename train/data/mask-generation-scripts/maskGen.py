from google.cloud import storage
import os
import pickle
import numpy as np
import json
from PIL import Image
from tqdm import tqdm_notebook, tnrange
#Get the labels stored in the pkl
HOME = "/home/pkhart/"
PKL_PATH = HOME + "dendritic-spines/train/data/mask-generation-scripts/allLabels.pkl"
# storage_client = storage.Client()
# bucket = storage_client.get_bucket("dendritic-spines")
# bucket.get_blob('jupyter/allLabels.pkl').download_to_filename(PKL_PATH)
# ids = json.loads(bucket.get_blob('layerTraining/fileIndexList.json').download_as_string().decode('utf-8'))


with open(PKL_PATH, 'rb') as file:
    data = pickle.load(file)
#Generate masks
MAX_HEIGHT = 20
MASK_PATH = HOME + "dendritic-spines/train/data/alternative-masks"

for n, tifStack in tqdm_notebook(enumerate(data), total=len(data)):
    maskStack = np.zeros(shape=[512,512,MAX_HEIGHT], dtype=bool)
    for spine in data[tifStack]['spines']:
        #Fill circle for spine mask
        x = (spine[0]+spine[2])/2.0
        y = (spine[1]+spine[3])/2.0
        radius = ((spine[0]-x)**2+(spine[1]-y)**2)**0.5
        for i in range(max(0 ,int(x - radius)), min(511, int(x + radius))):
            for j in range(max(0, int(y - radius)), min(511, int(y + radius))):
                if (x - i) ** 2 + (y - j) ** 2 < radius**2:
                    maskStack[i,j,spine[4]] = True
                    
    for layer in range(MAX_HEIGHT):
        fileName = tifStack[:-4]+"_"+str(layer)+'.png'
        if fileName[:-4] in ids:
            maskPath = os.path.join(MASK_PATH, fileName)
            if not os.path.exists(maskPath): #check if image already stored.
                with open(maskPath, 'w'): #create an empty file to store the image
                    pass
            #store the masks locally
            im = Image.fromarray(maskStack[:,:,layer])
            im.save(maskPath)
            print('saving to : ' + maskPath)

            
            
"""method used for getting the spine lables from the annotation files"""
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