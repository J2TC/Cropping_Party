import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


###############################################################################################
def use_Energy ():
    threshold = 30000;

    # Computing the energy of the error. 
    # Steps:
    #   1. Substract cropped and reference images
    #   2. Squaring of the absolute value of the difference
    #   3. Adding up all values of each pixel to get the final value of the energy





    return 0


def use_BitByBit ():






    return 0



###############################################################################################



# Loading image

reference = cv2.imread('reference.jpg',cv2.IMREAD_GRAYSCALE)

height_reference, width_reference = reference.shape

print('width = '+ str(width_reference))
print('height = '+ str(height_reference))

# Defining threshold by hand. This thresghold is an upperbound


n = 0
energy = []

# Loading the reference image for the processing

image = cv2.imread("Test.png", cv2.IMREAD_GRAYSCALE)
height_reference2, width_reference2 = image.shape


# Cropping cropping the original image with the dimensions of the reference

for col in tqdm(xrange(0, height_reference2-height_reference)):
    for row in xrange(0, width_reference2-width_reference):

        
        
        crop_img = image[col:col+height_reference, row: row + width_reference]         

        diff = reference-crop_img
        energy_diff = np.multiply(abs(diff),abs(diff))
            
        value_energy = 0
        
        for x in xrange(0, height_reference):
            for y in xrange(0, width_reference):
                value_energy += energy_diff[x,y]

        # Appending the data to plot it with matplotlib

        energy.append(value_energy)

        # If the value of the energy computed is less than the threshold, then we save the image

        if value_energy < threshold:
            cv2.imwrite("crop_results/crop%d.jpg" % n, crop_img)     # save frame as JPEG file
            n += 1
                    

# At the end we plot how the value of the energy was changing along the image, to estimate better our threshold

plt.plot(energy)
plt.ylabel('energy')
plt.show()
    





