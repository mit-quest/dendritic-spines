import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from random import random

def id_spines(mask):
    all_spine_pixels = []
    for t in range(mask.shape[3]):
        for z in range(mask.shape[2]):
            for y in range(mask.shape[1]):
                for x in range(mask.shape[0]):
                    if mask[x, y, z, t] == 1:
                        mask, spine_pixels = get_pixel_set(mask, tuple((x, y, z, t)))
                        all_spine_pixels.append(spine_pixels)
    return all_spine_pixels


def get_pixel_set(mask, start):
    this_spine_pixels = set()
    current_set = set()
    next_set = set()
    this_spine_pixels.add(start)
    current_set.add(start)
    changed = True
    while changed:
        for pixel in current_set:
            for x in range(max(0, pixel[0] - 1), min(pixel[0] + 2, mask.shape[0])):
                for y in range(max(0, pixel[1] - 1), min(pixel[1] + 2, mask.shape[1])):
                    for z in range(max(0, pixel[2] - 1), min(pixel[2] + 2, mask.shape[2])):
                        for t in range(max(0, pixel[3] - 1), min(pixel[3] + 2, mask.shape[3])):
                            if not mask[x, y, z, t] == 0:
                                next_set.add((x, y, z, t))
                                mask[x, y, z, t] = 0
        changed = len(current_set) > 0
        this_spine_pixels = this_spine_pixels.union(current_set)
        current_set = next_set
        next_set = set()
    return mask, this_spine_pixels


def get_spines_over_time(mouse, region, time=('a', 'c', 'e'), mask_directory="/home/pkhart/dendritic-spines/train/data/masks/", image_directory="/home/pkhart/dendritic-spines/train/data/images/"):
    depth = 20
    stack0 = np.zeros((512, 512, depth, len(time)), dtype=np.bool)
    images = np.zeros((512, 512, depth, len(time)), dtype=np.uint8)

    for t in range(len(time)):
        for layer in range(depth):
            stack0[:, :, layer, t] = np.asarray(Image.open(mask_directory + mouse + time[t] + region + "_" + str(layer) + ".png"))
            image_file_name = image_directory + mouse + time[t] + region + "_" + str(layer) + ".png"
            try:
                images[:, :, layer, t] = np.asarray(Image.open(image_file_name))
                
            except FileNotFoundError:
                print("can't find : " + image_file_name)
                depth = layer
                stack0 = stack0[:, :, :depth, :]
                images = images[:, :, :depth, :]
    spines = id_spines(stack0)  # TODO save this result in a text file?
    display_spines_over_time(spines, images, stack0, depth, time, mouse)
    


def display_spines_over_time(spines, images, masks, depth, time, mouse):
    stack_colored = np.zeros((512, 512, 3, depth, len(time)), dtype=np.uint8)

    size_tracker = np.zeros((len(spines), len(time)))
    spine_colors = np.zeros((len(spines), 3))
    for i in range(len(spines)):
        color = np.random.randint(low=50, high=255, size=3, dtype=np.uint8)  # generate random RGB color for each spine
        for (x, y, z, t) in spines[i]:
            stack_colored[x, y, :, z, t] = color
            size_tracker[i, t] += 1
        spine_colors[i] = color

    fig, ax = plt.subplots(nrows=len(time)*2, ncols=depth, figsize=(100,100))
    plt.gray()
    print(ax.shape)
    print(images.shape)
    for t in range(len(time)):
        for col in range(depth):
            print(2*t, col, depth, mouse)
            ax[2*t][col].axis("off")
            ax[2*t][col].imshow(images[:, :, col, t])
            if t == 0:
                ax[t][col].set_title("layer " + str(col))
    for t in range(len(time)):
        for col in range(depth):
            ax[2*t+1][col].axis("off")
            ax[2*t+1][col].imshow(stack_colored[:, :, :, col, t])
        
            
        

    fig.suptitle("Spines over time")
    fig.tight_layout()

#     plt.show()
    fig.savefig("./matplotlib/id_chart/"+mouse+".png", dpi=fig.dpi)
    
    fig2 = plt.figure()
#     fig2, ax2 = plt.subplots(2,2)
    for i in range(len(spines)):
        color = tuple(spine_colors[i][c] / 256 for c in range(3))
        x, y = [], []
        for t in range(len(size_tracker[i])):
            if not size_tracker[i][t] == 0:
                x.append(t)
                y.append(size_tracker[i][t])
        fig2.gca().plot(x, y, color=color, marker='o', linewidth=2, markersize=5)
    fig2.suptitle(str(len(spines)) + " unique spines identified over " + str(len(time)) + " weeks")
    fig2.savefig("./matplotlib/size_chart/"+mouse+".png", dpi=fig2.dpi)

#     plt.show()

# def display_spine_size_over_time(spines)
    
def id_spines_all_masks():
#     def get_spines_over_time(mouse, region, time=('a', 'c', 'e'), mask_directory="./", image_directory="../layerImages/"):

#     get_spines_over_time("ah1231", "002", mask_directory="/home/pkhart/dendritic-spines/train/data/inference/")
#     mask = np.zeros((512, 512, 10, 3))
#     time = ('a', 'c', 'e')

#     for t in range(len(time)):
#         for layer in range(10):
#             mask[:,:,layer,t] = np.array(Image.open("/home/pkhart/dendritic-spines/train/data/masks/ah1231"+time[t]+"002_"+str(layer)+".png"))
#     spxs = id_spines(mask)
#     print(spxs)
    
    
    
    
    
    
    time = ('a', 'c', 'e')
    mask_directory = "/home/pkhart/dendritic-spines/train/data/masks/"
    image_directory = "/home/pkhart/dendritic-spines/train/data/images/"
    inference_directory = "/home/pkhart/dendritic-spines/train/data/infer/"
    mouse_set = set()

    for r, d, f in os.walk(image_directory):
        for name in f:
            mouse_and_region = name[:6] + "," + name[7:10]
            mouse_set.add(mouse_and_region)
    print(mouse_set)
    for mouse_and_region in mouse_set:
        mouse, region = mouse_and_region.split(",")
        print("identifying spines for: " + mouse_and_region)
        get_spines_over_time(mouse, region, time, inference_directory, image_directory)


if __name__ == '__main__':
    id_spines_all_masks()