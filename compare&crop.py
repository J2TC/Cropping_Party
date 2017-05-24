import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


###############################################################################################
#   VARIABLES TO CONFIGURE THE SCRIPT

USE_ENERGY_APPROACH = True
USE_BITBYBIT_APPROACH = False
USE_SOBEL_X = True

PATH_FRAMES = "testsFrames/"
PATH_CROPS = "crop_results/"
PATH_REFERENCE = "references/"

NUMBER_REFS = 2                                 # This number is the first not to be included
NUMBER_FRAMES = 4                               # This number is the first not to be included

###############################################################################################
def use_Energy(ref, cropped):
    threshold = 0.9 * pow(10,9)
    sucess = False

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
            sucess = True

    return sucess,value_energy


def compareTwoPixels(PixelA, PixelB, epsilon):

    result = False

    if (PixelA==PixelB):
        result = True

    elif ((PixelA-epsilon) <= PixelB) and (PixelB <= (PixelA+epsilon)) :
        result = True

    return result

def use_BitByBit (ori, ref, cropped):

    minimumSimilarity = 0.9
    
    H, W = ref.shape
    nTotalPixels = H*W

    epsilon = 0
    nSimilarPixels = 0

    for x in xrange(0, H):
        for y in xrange(0,W):

            if compareTwoPixels(ref[x,y], cropped[x,y], epsilon):
                nSimilarPixels += 1

    similarity = float(nSimilarPixels) / float(nTotalPixels)

    if similarity >= minimumSimilarity:
        sucess = True

    return sucess,nSimilarPixels

def applySobelX(img):
    sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    abs_sobel = np.absolute(sobel)

    return abs_sobel



###############################################################################################

nConesDetected = 0
energy = []
similarity_results = []

for iteration in tqdm(xrange(1,NUMBER_FRAMES)):

    reference = cv2.imread(PATH_REFERENCE+"reference (1).png" ,cv2.IMREAD_GRAYSCALE)

    height_reference, width_reference = reference.shape

    original = cv2.imread(PATH_FRAMES+"Test (%d).jpg" % iteration, cv2.IMREAD_GRAYSCALE)

    if USE_SOBEL_X:
        image = applySobelX(original)
    else :
        image = original

    height_reference2, width_reference2 = image.shape


    for nReference in xrange(1, NUMBER_REFS):

        reference = cv2.imread(PATH_REFERENCE+"reference (%d).png" %  nReference,cv2.IMREAD_GRAYSCALE)

        if USE_SOBEL_X:
            reference = applySobelX(reference)

        for col in xrange(0, height_reference2-height_reference):
            for row in xrange(0, width_reference2-width_reference):
                crop_img = image[col:col+height_reference, row: row + width_reference]    

                if USE_ENERGY_APPROACH:     
                    success, value_energy = use_Energy(reference,crop_img)

                    if success:
                        cropped = original[col:col+height_reference, row: row + width_reference]  
                        cv2.imwrite(PATH_CROPS+"crop%d.png" % nConesDetected, cropped)     # save frame as PNG file
                        nConesDetected += 1


                    # Appending the data to plot it with matplotlib
                    energy.append(value_energy)

                if USE_BITBYBIT_APPROACH:
                    similarity = use_BitByBit(reference,crop_img)

                    if success:
                        cropped = original[col:col+height_reference, row: row + width_reference]  
                        cv2.imwrite(PATH_CROPS+"crop%d.png" % nConesDetected, cropped)     # save frame as PNG file
                        nConesDetected += 1
                    

                    # Appending the data to plot it with matplotlib
                    similarity_results.append(similarity)
                               

    # At the end we plot how the value of the energy was changing along the image, to estimate better our threshold

#    if USE_ENERGY_APPROACH: 
#        plt.plot(energy)
#        plt.ylabel('energy')
#        plt.show()
        
#    if USE_BITBYBIT_APPROACH:
#        plt.plot(similarity_results)
#        plt.ylabel('similar pixels')
#        plt.show()
