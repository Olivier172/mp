import numpy as np

def main():
    a=np.zeros(10,dtype="int")
    print(a)
    b=np.zeros((5,10),dtype="int")
    #print(b)
    for i in range(len(b)):
        for j in range(len(b[0])):
            print(b[i][j],end=" ")
        print(" ")
        
main()