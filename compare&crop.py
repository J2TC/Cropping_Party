import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


###############################################################################################
#   VARIABLES TO CONFIGURE THE SCRIPT

USE_ENERGY_APPROACH = False
USE_BITBYBIT_APPROACH = False





###############################################################################################
def use_Energy(ref, cropped, n):
    threshold = 30000;

    # Computing the energy of the error. 
    # Steps:
    #   1. Substract cropped and reference images
    #   2. Squaring of the absolute value of the difference
    #   3. Adding up all values of each pixel to get the final value of the energy

    diff = ref-cropped
    energy_diff = np.multiply(abs(diff),abs(diff))
            
    value_energy = 0
        
    for x in xrange(0, height_reference):
        for y in xrange(0, width_reference):
            value_energy += energy_diff[x,y]


    # If the value of the energy computed is less than the threshold, then we save the image

    if value_energy < threshold:
            cv2.imwrite("crop_results/crop%d.png" % n, cropped)     # save frame as PNG file
            n += 1

    return value_energy, n


def compareTwoPixels(PixelA, PixelB, epsilon):

    result = False

    if (PixelA==PixelB):
        result = True

    elif ((PixelA-epsilon) <= PixelB) and (PixelB <= (PixelA+epsilon)) :
        result = True

    return result

def use_BitByBit (ref, cropped, n):
    minimumSimilarity = 0.8
    
    H, W = ref.shape
    nTotalPixels = H*W

    epsilon = 0
    nSimilarPixels = 0

    for x in xrange(0, H):
        for y in xrange(0,W):

            if compareTwoPixels(ref(x,y), cropped(x,y), epsilon):
                nSimilarPixels += 1

    similarity = nSimilarPixels / nTotalPixels

    if similarity >= minimumSimilarity:
        cv2.imwrite("crop_results/crop%d.png" % n, cropped)     # save frame as PNG file
        n += 1


    return similarity, n





###############################################################################################



# Loading image

reference = cv2.imread('reference.jpg',cv2.IMREAD_GRAYSCALE)

height_reference, width_reference = reference.shape

print('width = '+ str(width_reference))
print('height = '+ str(height_reference))

# Defining threshold by hand. This thresghold is an upperbound

nConesDetected = 0
energy = []
similarity_results = []

# Loading the reference image for the processing

image = cv2.imread("Test.png", cv2.IMREAD_GRAYSCALE)
height_reference2, width_reference2 = image.shape


# Cropping cropping the original image with the dimensions of the reference

for col in tqdm(xrange(0, height_reference2-height_reference)):
    for row in xrange(0, width_reference2-width_reference):

        crop_img = image[col:col+height_reference, row: row + width_reference]    


        if USE_ENERGY_APPROACH:     
            value_energy, nConesDetected = use_Energy(reference,crop_img, nConesDetected)
            # Appending the data to plot it with matplotlib
            energy.append(value_energy)

        if USE_BITBYBIT_APPROACH:
            similarity, nConesDetected = use_BitByBit(reference,crop_img, nConesDetected)
            # Appending the data to plot it with matplotlib
            similarity_results.append(similarity)


                       

# At the end we plot how the value of the energy was changing along the image, to estimate better our threshold

plt.plot(energy)
plt.ylabel('energy')
plt.show()
    





