from PIL import Image
from network import NETWORK
#cat.jpg est l'image d'entraînement
def Resize(path):
    img = Image.open(path)
    simg = img.resize((15, 15))
    simg.save(path)
def SetInputsForNetwork(path):
    Resize(path)
    img = Image.open(path)
    pix_val = list(img.getdata())
    img.close()
    ToReturn = []
    for i in range(0, len(pix_val)): ToReturn.append(float((pix_val[i][0] + pix_val[i][1] + pix_val[i][2]) / 300))
    return ToReturn
N = NETWORK(SetInputsForNetwork('cat.jpg'), 5, 1, 0.04, 0, AcceptError=0.001, sigmoide=1)
N.Train()
print(N.Play(SetInputsForNetwork('cat.jpg')))
print(N.Play(SetInputsForNetwork('pencil.jpg')))