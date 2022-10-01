#a file to try out pytorch library functions and basics
import torch 
import numpy as np

def basics():
    print("Testing out the basics of pytorch lib")
    print("Tensors:")

    data = [[1,2],[3,4]]
    x_data = torch.tensor(data)
    print(f"Tensor from a nested list: \n {x_data} \n")

    datanp = np.array(data)
    y_data = torch.from_numpy(datanp)
    print(f"Tensor from a numpy array \n {y_data} \n")

    z_data= torch.rand_like(x_data,dtype=float) #random tensor met dezelfde vorm van array als x_data
    print(f"Tensor from a another tensor \n {z_data} \n")

    #ge kunt de shape maken in een tuple
    shape = (4,8)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape,dtype=int)
    zeros_tensor = torch.zeros(shape)
    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")

    print("attributes van tensors")
    tensor = torch.rand(3,2)
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")

    if torch.cuda.is_available():
        print("yes")
        #tensor=tensor.to("cuda") #dit geeft toch een runtime error :/ ik weet niet hoe ik nvidia drivers op linux moet zetten
    else:
        print("sad")
    

    t = torch.zeros((10,10),dtype=int)
    for i in range(10):
        for j in range(10):
            t[i][j]=i+j
    print(t) #tensors werken dus als gewone arrays maar ge kunt er sneller operaties op doen op GPU bv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")






def main():
    basics()
    return 0

main()