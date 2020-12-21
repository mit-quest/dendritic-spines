Dendritic spines image segmentation 

To run the code on satori
1) Queue and launch a jupyter lab satori job
2) clone the github repository ($ git clone https://github.com/mit-quest/dendritic-spines.git)
3) CD into the directory ($ cd dendritic-spines)
3) Initialize the conda environment ($ conda create --name <den-spines> --file requirements.txt)
    - This step may take a long time to run
4) Create and download a GCP access key json for the data bucket (https://console.cloud.google.com/storage/browser/dendritic-spines)
5) run the training script ($ python ./train/unet.py).
    -This will take a long time the first time you run it since it has to first download the images and masks
    -Subsequent runs may also take a long time depending on the training paramaters
    
  
TODO:
1) Explore different mask generation algorithms. The script dendritic-spines/train/data/mask-generation-scripts/alt-mask-gen.py contains some code that we previously used to interpolate masks for unlabled spines in neigboring layers of the tiff.
2) Tweak unet training paramaters to improve inference

    
