# Importing Image class from PIL module
from PIL import Image
 
# Opens a image in RGB mode
im = Image.open(r"test_0.jpeg")
 
# Size of the image in pixels (size of original image)
# (This is not mandatory)
width, height = im.size
print("width is " + str(width))
print("height is " + str(height))

#reading anotations
f = open("test_0.txt", "r")

i=0
while(True):
    lijn = f.readline()
    if lijn == " ":
        break
    #print("Eerste lijn in dit annotatiebestand")
    #print(lijn)
    annotaties = lijn.split()
    #print(annotaties)
    x_center = float(annotaties[1])
    y_center = float(annotaties[2])
    width_a = float(annotaties[3])
    height_a = float(annotaties[4])
    print("x_center is " + str(x_center) + " y_center is " + str(y_center) + " width_a " + str(width_a) + " height_a " + str(height_a))
 
    # Setting the points for cropped image
    # x_center y_center width height
    left = int(x_center*width -(width_a*width)/2)
    top = int(y_center*height +(height_a*height)/2)
    right = int(x_center*width +(width_a*width)/2)
    bottom = int(y_center*height -(height_a*height)/2)
    print("left is " + str(left) + " right is " + str(right) + " bottem " + str(bottom) + " top " + str(top))
 
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, bottom, right, top))
 
    # Shows the image in image viewer
    #im1.show()
    i+=1
    im1.save("result/img" + str(i) + ".jpeg")
    

f.close()

