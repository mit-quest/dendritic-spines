import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


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


def get_spines_over_time(mouse, region, time=('a', 'c', 'e'), mask_directory="./", image_directory="../layerImages/"):
    depth = 20
    stack0 = np.zeros((512, 512, depth, len(time)), dtype=np.bool)
    images = np.zeros((512, 512, depth, len(time)), dtype=np.uint8)

    for t in range(len(time)):
        for layer in range(depth):
            try:
                stack0[:, :, layer, t] = np.asarray(
                    Image.open(mask_directory + mouse + time[t] + region + "_" + str(layer) + ".png"))
                images[:, :, layer, t] = np.asarray(
                    Image.open(image_directory + mouse + time[t] + region + "_" + str(layer) + ".png"))
            except FileNotFoundError:
                depth = layer
                stack0 = stack0[:, :, :depth, :]
                images = images[:, :, :depth, :]

    spines = id_spines(stack0)  # TODO save this result in a text file?
    display_spines_over_time(spines, images, stack0, depth, time)


def display_spines_over_time(spines, images, masks, depth, time):
    stack_colored = np.zeros((512, 512, 3, depth, len(time)), dtype=np.uint8)

    size_tracker = np.zeros((len(spines), len(time)))
    spine_colors = np.zeros((len(spines), 3))
    for i in range(len(spines)):
        color = np.random.randint(low=50, high=255, size=3, dtype=np.uint8)  # generate random RGB color for each spine
        for (x, y, z, t) in spines[i]:
            stack_colored[x, y, :, z, t] = color
            size_tracker[i, t] += 1
        spine_colors[i] = color

    fig, ax = plt.subplots(nrows=len(time), ncols=depth, figsize=(100, 100))
    plt.gray()

    for t in range(len(time)):

        for col in range(depth):
            ax[t][col].axis("off")
            if col >= 3:
                ax[t][col].imshow(stack_colored[:, :, :, col, t])

                if t == 0:
                    ax[t][col].set_title("layer " + str(col))

    fig.suptitle("Spines over time")

    plt.show()

    for i in range(len(spines)):
        color = tuple(spine_colors[i][c] / 256 for c in range(3))
        x, y = [], []
        for t in range(len(size_tracker[i])):
            if not size_tracker[i][t] == 0:
                x.append(t)
                y.append(size_tracker[i][t])
        plt.plot(x, y, color=color, marker='o', linewidth=2, markersize=5)
        plt.title(str(len(spines)) + " unique spines identified over " + str(len(time)) + " weeks")
    plt.show()


def id_spines_all_masks():
    time = ('a', 'c', 'e')
    mask_directory = "./"
    image_directory = "../layerImages/"
    mouse_set = set()

    for r, d, f in os.walk(image_directory):
        for name in f:
            mouse_and_region = name[:6] + "," + name[7:10]
            mouse_set.add(mouse_and_region)

    for mouse_and_region in mouse_set:
        mouse, region = mouse_and_region.split(",")
        print("identifying spines for: " + mouse_and_region)
        get_spines_over_time(mouse, region, time, mask_directory, image_directory)


if __name__ == '__main__':
    id_spines_all_masks()
