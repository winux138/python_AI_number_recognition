from PIL import Image
import numpy as np


def img_out(npArray, filename):
    # on converti en image
    img = Image.fromarray(npArray)
    # on enregistre l'image
    img.save(filename)
