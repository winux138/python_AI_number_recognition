from PIL import Image
import numpy as np


def img_out(npArray, filename):
    # on recupere les infos sur les dimensions de l'image
    img_h, img_w = npArray.shape[0], npArray.shape[1]
    # on cree une image vierge avec ces dimensions
    data = np.zeros((img_h, img_w), dtype=np.uint8)
    # on remplie cette image avec nos donnees
    for i in range(0, img_h):
        for j in range(0, img_w):
            data[i][j] = npArray[i][j]
    # on converti en image
    img = Image.fromarray(data)
    # on enregistre l'image
    img.save(filename)
