from pathlib import Path
from PIL import Image
cropcounter=0 #global var to count crops

def main():
    print("Main function imagecrop.py")

    # structure of the dataset $tree -L 3
    # └── SKU110K_fixed
    #       ├── images
    #       │   ├── test
    #       │   ├── train
    #       │   └── val
    #       └── labels
    #           ├── test
    #           ├── train
    #           └── val
    dir = Path("/home/olivier/Documents/mp/SKU110K/SKU110K_fixed") #absolute path to the dataset on disk
    #expected structure where the files will be saved
    # cropped_images
    #     ├── test
    #     ├── train
    #     └── val
    dirResult = Path("/home/olivier/Documents/mp/cropped_images") #path to where the crops will be saved
    
    # the test pictures
    global cropcounter
    cropcounter = 0 #resetting the cropcounter to 0
    logfile_test = open(Path("logfile_test.txt"),"w")
    for i in range(0,2941):
        filenameImage= dir.joinpath("images","test", "test_"+ str(i) + ".jpg") 
        filenameLabels= dir.joinpath("labels","test", "test_"+ str(i) + ".txt") 
        baseFilenameResult= dirResult.joinpath("test") 
        filenameResult = "testcrop_"  # "baseFilenameResult/testcrop_cropcounter" 
        rv = cropImage(filenameImage,filenameLabels,baseFilenameResult,filenameResult)
        if(rv==-1):
            print("something went wrong while excecuting cropImage")
            print("filenameImage " + str(filenameImage))
            print("filenameLabels " + str(filenameLabels))
            logfile_test.write("something went wrong while excecuting cropImage\n")
            logfile_test.write("filenameImage " + str(filenameImage) +"\n")
            logfile_test.write("filenameLabels " + str(filenameLabels) +"\n")
        else:
            print("Excecution cropImage OK :" + str(filenameImage))
    logfile_test.close()
   
    #the train pictures
    global cropcounter
    cropcounter = 0 #resetting the cropcounter to 0
    logfile_train = open(Path("logfile_train.txt"),"w")
    for i in range(0,8235):
        filenameImage= dir.joinpath("images","train","train_"+ str(i) + ".jpg") 
        filenameLabels= dir.joinpath("labels","train","train_"+ str(i) + ".txt")  
        baseFilenameResult= dirResult.joinpath("train") 
        filenameResult = "traincrop_"  # "baseFilenameResult/traincrop_cropcounter" 
        rv = cropImage(filenameImage,filenameLabels,baseFilenameResult,filenameResult)
        if(rv==-1):
            print("something went wrong while excecuting cropImage")
            print("filenameImage " + str(filenameImage))
            print("filenameLabels " + str(filenameLabels))
            logfile_train.write("something went wrong while excecuting cropImage\n")
            logfile_train.write("filenameImage " + str(filenameImage) +"\n")
            logfile_train.write("filenameLabels " + str(filenameLabels) +"\n")
        else:
            print("Excecution cropImage OK :" + str(filenameImage))
    logfile_train.close()

    #the val pictures
    global cropcounter
    cropcounter = 0 #resetting the cropcounter to 0
    logfile_val = open(Path("logfile_val.txt"),"w")
    for i in range(0,600): #600
        filenameImage= dir.joinpath("images","val", "val_"+ str(i) + ".jpg") 
        filenameLabels= dir.joinpath("labels", "val", "val_"+ str(i) + ".txt") 
        baseFilenameResult= dirResult.joinpath("val") 
        filenameResult = "valcrop_"  # "baseFilenameResult/valcrop_cropcounter" 
        rv = cropImage(filenameImage,filenameLabels,baseFilenameResult,filenameResult)
        if(rv==-1):
            print("something went wrong while excecuting cropImage")
            print("filenameImage " + str(filenameImage))
            print("filenameLabels " + str(filenameLabels))
            logfile_val.write("something went wrong while excecuting cropImage\n")
            logfile_val.write("filenameImage " + str(filenameImage) +"\n")
            logfile_val.write("filenameLabels " + str(filenameLabels) +"\n")
        else:
            print("Excecution cropImage OK :" + str(filenameImage))
    logfile_val.close()
    
    return 0


# @param filenameImage : the path to the input image
# @param filenameLabels : path to the corresponding labels of that input image, conataining bounding box coordinates of the crops
# @param baseFilenameResult : path to the resulting files (crops) to be saved
# @param filenameResult : name of the file to be saved
# returns 0 if the image was succesfully cropped, -1 if an error occured
def cropImage(filenameImage:Path,filenameLabels:Path,baseFilenameResult:Path, filenameResult):
    try:
        im = Image.open(filenameImage,"r")# Opens a image in RGB mode
    except:
        print("Something went wrong openening this image " + str(filenameImage))
        return -1
 
    # Size of the image in pixels (size of original image)
    width, height = im.size
    #print("width is " + str(width))
    #print("height is " + str(height))

    try:
        f = open(filenameLabels, "r") #reading anotations
    except:
        print("Something went wrong openening this annotation file " + str(filenameLabels))
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
        bottom = int(y_center*height + (height_a*height)/2)
        top = int(y_center*height - (height_a*height)/2)
        right = int(x_center*width +(width_a*width)/2)
        #print("left is " + str(left) + " right is " + str(right) + " bottem " + str(bottom) + " top " + str(top))
        # Cropped image of above dimension
        # (It will not change original image)
        im1 = im.crop((left, top, right, bottom))
        # Shows the image in image viewer
        #im1.show()
        global cropcounter 
        filename_result = baseFilenameResult.joinpath(filenameResult + str(cropcounter) + ".jpg") 
        cropcounter+=1
        #print("resulting file " + filename_result)
        try:
            im1.save(filename_result)
        except:
            print("Something went wrong saving  " + str(filename_result))
            return -1

    im.close()
    f.close()
    return 0 # everything went well, all the crops were cut out and saved

if __name__ == "__main__":
    main()