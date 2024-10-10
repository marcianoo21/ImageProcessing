from PIL import Image
import numpy as np
import sys


im = Image.open("lenac.bmp")

arr = np.array(im.getdata())
if arr.ndim == 1: 
    numColorChannels = 1
    arr = arr.reshape(im.size[1], im.size[0])
else:
    numColorChannels = arr.shape[1]
    arr = arr.reshape(im.size[1], im.size[0], numColorChannels)

# newIm = Image.fromarray(arr.astype(np.uint8))
# new_size = (255, 255)
# newIm = newIm.resize(new_size)
# newIm.save("result.bmp")




def doBrightness(param, arr):
    print("Function doBrightness invoked with param: " + str(param))
    arr += int(param)
    arr[arr > 255] = 255  
    arr[arr < 0] = 0 

def doContrast(param, arr):
    print("Function doContrast invoked with param: " + param)
    arr = (arr - 128) * float(param) + 128
    arr[arr > 255] = 255  
    arr[arr < 0] = 0  
    return arr
    

def doNegative(arr):
    print("Negative action")
    arr = 255 - arr
    arr[arr > 255] = 255  
    arr[arr < 0] = 0  
    return arr


def doDefault(arr):
    print("Default action")
    return arr


def doVerticalFlip(arr):
    print("Vertical flip action")
    arr = arr[::-1]
    return arr


def doHorizontalFlip(arr):
    print("Horizontal flip action")
    arr = arr[:, ::-1]
    return arr

def doDiagonalFlip(arr):  
    print("Diagonal flip action")
    arr = arr[::-1, ::-1]
    return arr

if len(sys.argv) == 1:
    print("No command line parameters given.\n")
    sys.exit()

command = sys.argv[1]

if len(sys.argv) == 2:
    if command == '--negative':
        arr = doNegative(arr)
    elif command == '--default':
        arr = doDefault(arr)
    elif command == '--vflip':
        arr = doVerticalFlip(arr)
    elif command == '--hflip':
        arr = doHorizontalFlip(arr)
    elif command == '--dflip':
        arr = doDiagonalFlip(arr)
    else:
        print("Too few command line parameters given.\n")
        sys.exit()
else:
    param = sys.argv[2]
    if command == '--brightness':
        arr = doBrightness(param, arr)
    elif command == '--contrast':
        arr = doContrast(param, arr)
    else:
        print("Unknown command: " + command)
        sys.exit()


newIm = Image.fromarray(arr.astype(np.uint8))
newIm.save("result.bmp")