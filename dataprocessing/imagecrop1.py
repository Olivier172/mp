# imagecrop1.py 
# alternative version of imagecrop.py
# this script saves the image crops from the same image 
# in different folders within the test train val folders

from pathlib import Path
from PIL import Image
imageCounter = 0 #global variable to count which image we are cropping from the test, train or val set

def main():
    print("Main function imagecrop1.py")

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
    # expected structure where the files will be saved
    # cropped_images1
    #     ├── test
    #     ├── train
    #     └── val
    dirResult = Path("/home/olivier/Documents/mp/cropped_images1") #path to where the crops will be saved
    
    # the test pictures
    global imageCounter
    imageCounter=0 #count the test images
    logfile_test = open(Path("logfile_test1.txt"),"w")
    for i in range(0,2941):
        filepathImage= dir.joinpath("images","test", "test_"+ str(i) + ".jpg") 
        filepathLabels= dir.joinpath("labels","test", "test_"+ str(i) + ".txt") 
        baseFilepathResult= dirResult.joinpath("test") 
        filenameResult = "testcrop_"  # "baseFilepathResult/testcrop_cropcounter" 
        rv = cropImage(filepathImage,filepathLabels,baseFilepathResult,filenameResult)
        if(rv==-1):
            print("something went wrong while excecuting cropImage")
            print("filepathImage " + str(filepathImage))
            print("filepathLabels " + str(filepathLabels))
            logfile_test.write("something went wrong while excecuting cropImage\n")
            logfile_test.write("filepathImage " + str(filepathImage) +"\n")
            logfile_test.write("filepathLabels " + str(filepathLabels) +"\n")
        else:
            print("Excecution cropImage OK :" + str(filepathImage))
    logfile_test.close()
   
    #the train pictures
    imageCounter=0 #count the train images
    logfile_train = open(Path("logfile_train1.txt"),"w")
    for i in range(0,8235):
        filepathImage= dir.joinpath("images","train","train_"+ str(i) + ".jpg") 
        filepathLabels= dir.joinpath("labels","train","train_"+ str(i) + ".txt")  
        baseFilepathResult= dirResult.joinpath("train") 
        filenameResult = "traincrop_"  # "baseFilepathResult/traincrop_cropcounter" 
        rv = cropImage(filepathImage,filepathLabels,baseFilepathResult,filenameResult)
        if(rv==-1):
            print("something went wrong while excecuting cropImage")
            print("filepathImage " + str(filepathImage))
            print("filepathLabels " + str(filepathLabels))
            logfile_train.write("something went wrong while excecuting cropImage\n")
            logfile_train.write("filepathImage " + str(filepathImage) +"\n")
            logfile_train.write("filepathLabels " + str(filepathLabels) +"\n")
        else:
            print("Excecution cropImage OK :" + str(filepathImage))
    logfile_train.close()

    #the val pictures
    imageCounter=0 #count the val images
    logfile_val = open(Path("logfile_val1.txt"),"w")
    for i in range(0,600): #600
        filepathImage= dir.joinpath("images","val", "val_"+ str(i) + ".jpg") 
        filepathLabels= dir.joinpath("labels", "val", "val_"+ str(i) + ".txt") 
        baseFilepathResult= dirResult.joinpath("val") 
        filenameResult = "valcrop_"  # "baseFilepathResult/valcrop_cropcounter" 
        rv = cropImage(filepathImage,filepathLabels,baseFilepathResult,filenameResult)
        if(rv==-1):
            print("something went wrong while excecuting cropImage")
            print("filepathImage " + str(filepathImage))
            print("filepathLabels " + str(filepathLabels))
            logfile_val.write("something went wrong while excecuting cropImage\n")
            logfile_val.write("filepathImage " + str(filepathImage) +"\n")
            logfile_val.write("filepathLabels " + str(filepathLabels) +"\n")
        else:
            print("Excecution cropImage OK :" + str(filepathImage))
    logfile_val.close()
    
    return 0


# @param filepathImage : the path to the input image
# @param filepathLabels : path to the corresponding labels of that input image, conataining bounding box coordinates of the crops
# @param baseFilepathResult : path to the resulting files (crops) to be saved
# @param filenameResult : name of the file to be saved
# returns 0 if the image was succesfully cropped, -1 if an error occured
def cropImage(filepathImage:Path,filepathLabels:Path,baseFilepathResult:Path, filenameResult:str):
    global imageCounter

    try:
        im = Image.open(filepathImage,"r")# Opens a image in RGB mode
    except:
        print("Something went wrong openening this image " + str(filepathImage))
        imageCounter+=1
        return -1
 
    # Size of the image in pixels (size of original image)
    width, height = im.size
    #print("width is " + str(width))
    #print("height is " + str(height))

    try:
        f = open(filepathLabels, "r") #reading anotations
    except:
        print("Something went wrong openening this annotation file " + str(filepathLabels))
        imageCounter+=1
        return -1

    cropcounter = 0 #resetting the cropcounter to 0 for crops from a new image
    #creating a dir for this image and it's crops
    path = baseFilepathResult / ("im" + str(imageCounter)) 
    path.mkdir(exist_ok=True)
    baseFilepathResult = path
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
        filename_result = baseFilepathResult.joinpath("im" + str(imageCounter) + "_crop" + str(cropcounter) + ".jpg") 
        cropcounter+=1
        #print("resulting file " + filename_result)
        try:
            im1.save(filename_result)
        except:
            print("Something went wrong saving  " + str(filename_result))
            imageCounter+=1
            return -1
    imageCounter+=1
    im.close()
    f.close()
    return 0 # everything went well, all the crops were cut out and saved

if __name__ == "__main__":
    main()