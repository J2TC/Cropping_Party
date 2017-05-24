import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


###############################################################################################
#   VARIABLES TO CONFIGURE THE SCRIPT

SHOW_GRAPHS = True
ENABLE_SAVE_CROPS = True

PATH_FRAMES = "testsFrames/"
PATH_CROPS = "crop_results/"
PATH_REFERENCE = "references/"

NUMBER_REFS = 2                                 # This number is the first not to be included
NUMBER_FRAMES = 2                               # This number is the first not to be included

###############################################################################################
def use_Energy_Sobel(ref, cropped):
    threshold = 1.0 * pow(10,9)

    sucess = False

    diff = ref-cropped
    energy_diff = np.multiply(abs(diff),abs(diff))
            
    value_energy = 0
        
    for x in xrange(0, height_reference):
        for y in xrange(0, width_reference):
            value_energy += energy_diff[x,y]

    if value_energy < threshold:
            sucess = True

    return sucess,value_energy

def use_Energy_Canny(ref, cropped):
    threshold = 150

    sucess = False

    diff = ref-cropped
    energy_diff = np.multiply(abs(diff),abs(diff))
            
    value_energy = 0
        
    for x in xrange(0, height_reference):
        for y in xrange(0, width_reference):
            value_energy += energy_diff[x,y]

    if value_energy < threshold:
            sucess = True

    return sucess,value_energy


def applySobelX(img):
    sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    abs_sobel = np.absolute(sobel)

    return abs_sobel

def applyCanny(img):
    canny = cv2.Canny(img,100,200)

    return canny



###############################################################################################

nConesDetected = 0
energy_Sobel = []
energy_Canny = []

original = cv2.imread(PATH_FRAMES+"Test (3).jpg", cv2.IMREAD_GRAYSCALE)

image_Sobel = applySobelX(original)
image_Canny = applyCanny(original)
    
height_reference2, width_reference2 = original.shape


reference = cv2.imread(PATH_REFERENCE+"reference (1).png",cv2.IMREAD_GRAYSCALE)
height_reference, width_reference = reference.shape
reference_Sobel = applySobelX(reference)
reference_Canny = applyCanny(reference)

for col in tqdm(xrange(0, height_reference2-height_reference)):
    for row in xrange(0, width_reference2-width_reference):

        crop_img_Sobel = image_Sobel[col:col+height_reference, row: row + width_reference]    
        crop_img_Canny = image_Canny[col:col+height_reference, row: row + width_reference]

        success_Sobel, value_energy_Sobel = use_Energy_Sobel(reference_Sobel,crop_img_Sobel)
        success_Canny, value_energy_Canny = use_Energy_Canny(reference_Canny,crop_img_Canny)

        if success_Sobel & success_Canny & ENABLE_SAVE_CROPS:
            cropped = original[col:col+height_reference, row: row + width_reference]  
            cv2.imwrite(PATH_CROPS+"crop%d.png" % nConesDetected, cropped)     
            nConesDetected += 1

        # Appending the data to plot it with matplotlib
        energy_Sobel.append(value_energy_Sobel)                               
        energy_Canny.append(value_energy_Canny) 

if  SHOW_GRAPHS: 
    plt.plot(energy_Sobel)
    plt.ylabel('energy')
    plt.title('Sobel')
    plt.show()
    plt.plot(energy_Canny)
    plt.ylabel('energy')
    plt.title('Canny')
    plt.show()
        
