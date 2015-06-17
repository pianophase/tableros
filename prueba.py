from scipy.cluster.vq import vq, kmeans, whiten
from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

# aplicar filtro gaussiano sigma=1.2

im  = misc.imread('prueba.jpg')
im1 = ndimage.filters.gaussian_filter(im,2)
im2 = im.reshape((im.shape[0]*im.shape[1],3))

misc.imshow(im1)
