# Importing Image class from PIL module
from PIL import Image
cropcounter=0 #global var to count crops

def main():
    print("Main function imagecrop.py")
    dir= "E:\AAA_MASTERPROEF\github\sku110k_dataset\SKU110K_fixed" #absolute path to the dataset on the disk, change this on another PC!

    #the test pictures
    logfile_test = open("logfile_test.txt","w")
    for i in range(0,2941):
        filenameImage= dir +"\images\\test\\test_"+ str(i) + ".jpg" #double \ is needed before t because otherwise python thinks you mean tab as \t
        filenameLabels= dir + "\labels\\test\\test_"+ str(i) + ".txt"
        baseFilenameResult="cropped_images\\test\\testcrop_"

        rv = cropImage(filenameImage,filenameLabels,baseFilenameResult)
        if(rv==-1):
            print("something went wrong while excecuting cropImage")
            print("filenameImage " + filenameImage)
            print("filenameLabels " + filenameLabels)
            logfile_test.write("something went wrong while excecuting cropImage\n")
            logfile_test.write("filenameImage " + filenameImage +"\n")
            logfile_test.write("filenameLabels " + filenameLabels +"\n")
        else:
            print("Excecution cropImage OK :" + filenameImage)
    logfile_test.close()
   
  
    #the train pictures
    global cropcounter
    cropcounter = 0 #resetting the cropcounter to 0
    logfile_train = open("logfile_train.txt","w")
    for i in range(0,8235):
        filenameImage= dir +"\images\\train\\train_"+ str(i) + ".jpg" #double \ is needed before t because otherwise python thinks you mean tab as \t
        filenameLabels= dir + "\labels\\train\\train_"+ str(i) + ".txt"
        baseFilenameResult="cropped_images\\train\\traincrop_"

        rv = cropImage(filenameImage,filenameLabels,baseFilenameResult)
        if(rv==-1):
            print("something went wrong while excecuting cropImage")
            print("filenameImage " + filenameImage)
            print("filenameLabels " + filenameLabels)
            logfile_train.write("something went wrong while excecuting cropImage\n")
            logfile_train.write("filenameImage " + filenameImage +"\n")
            logfile_train.write("filenameLabels " + filenameLabels +"\n")
        else:
            print("Excecution cropImage OK :" + filenameImage)
    logfile_train.close()
    
    #the val pictures
    global cropcounter
    cropcounter = 0 #resetting the cropcounter to 0
    logfile_val = open("logfile_val.txt","w")
    for i in range(0,600):
        filenameImage= dir +"\images\\val\\val_"+ str(i) + ".jpg" #double \ is needed before t because otherwise python thinks you mean tab as \t
        filenameLabels= dir + "\labels\\val\\val_"+ str(i) + ".txt"
        baseFilenameResult="cropped_images\\val\\valcrop_"

        rv = cropImage(filenameImage,filenameLabels,baseFilenameResult)
        if(rv==-1):
            print("something went wrong while excecuting cropImage")
            print("filenameImage " + filenameImage)
            print("filenameLabels " + filenameLabels)
            logfile_val.write("something went wrong while excecuting cropImage\n")
            logfile_val.write("filenameImage " + filenameImage +"\n")
            logfile_val.write("filenameLabels " + filenameLabels +"\n")
        else:
            print("Excecution cropImage OK :" + filenameImage)
    logfile_val.close()
    

    return 0


# @param filenameImage : the input image
# @param filenameLabels : corresponding labels of that image that conatain bounding box coords of the crops
# @param baseFilenameResult : base name of the resulting files (crops)
#returns if the image was succesfully cropped
def cropImage(filenameImage,filenameLabels,baseFilenameResult):
    try:
        im = Image.open(filenameImage,"r")# Opens a image in RGB mode
    except:
        print("Something went wrong openening this image " + filenameImage)
        return -1
 
    # Size of the image in pixels (size of original image)
    width, height = im.size
    #print("width is " + str(width))
    #print("height is " + str(height))

    try:
        f = open(filenameLabels, "r") #reading anotations
    except:
        print("Something went wrong openening this annotation file " + filenameLabels)
        return -1

    while(True):
        line = f.readline() #format= class x_center y_center width_a height_a
        annotations = line.split()
        if(len(annotations) ==0): #if line is empty, len of annotations will be zero
            break #stop this loop, all bounding boxes of this image have been cropped
        #annotations[0] -> class , not needed here
        x_center = float(annotations[1])
        y_center = float(annotations[2])
        width_a = float(annotations[3]) #width of the annotated bounding box
        height_a = float(annotations[4]) #height of the annotated bounding box
        #print("x_center is " + str(x_center) + " y_center is " + str(y_center) + " width_a " + str(width_a) + " height_a " + str(height_a))
        # Setting the points for cropped image
        # x_center y_center width_a height_a -> left bottem right top
        left = int(x_center*width -(width_a*width)/2)
        bottom = int(y_center*height -(height_a*height)/2)
        top = int(y_center*height +(height_a*height)/2)
        right = int(x_center*width +(width_a*width)/2)
        #print("left is " + str(left) + " right is " + str(right) + " bottem " + str(bottom) + " top " + str(top))
        # Cropped image of above dimension
        # (It will not change original image)
        im1 = im.crop((left, bottom, right, top))
        # Shows the image in image viewer
        #im1.show()
        global cropcounter 
        filename_result = baseFilenameResult + str(cropcounter) + ".jpg"
        cropcounter+=1
        #print("resulting file " + filename_result)
        try:
            im1.save(filename_result)
        except:
            print("Something went wrong saving  " + filename_result)
            return -1

    im.close()
    f.close()
    return 0

#calling main function
main()