from scipy.cluster.vq import vq, kmeans, whiten

from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

# aplicar filtro gaussiano sigma=1.2

im  = misc.imread('prueba.jpg')
im1 = ndimage.filters.gaussian_filter(im,2)
im2 = im.reshape((im.shape[0]*im.shape[1],3))

misc.imshow(im1)

# Cluster
whitened            = whiten(im2)
[code_book, distor] = kmeans(whitened, 3) 
[imc, imd]           = vq(whitened, code_book)
final               = im.reshape(((im.shape[0], im.shape[1])))
misc.imshow(final)

# Sobel
sx = ndimage.sobel(im, axis=0, mode='constant')
sy = ndimage.sobel(im, axis=1, mode='constant')
sob = np.hypot(sx, sy)

# Algo
labels, numObjects = scipy.ndimage.label(final) 

#find the number of pixels in each object (alternatively use ndimage.measure)
nPixels, bins = scipy.histogram(labels, scipy.arange(numObjects) + 1)

plt.plot(nPixels)

newMask = scipy.zeros(labels.shape)

#loop over objects
for i in range(numObjects):
    #and add back into mask if larger than the cutoff
    if nPixels[i] > sizeCutoff:
        newMask += (labels == (i+1))





hist, bin_edges = np.histogram(im, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

aux1 = im > 100

binary_img = im < 200

plt.imshow(binary_img, cmap = plt.cm.gray, interpolation = 'nearest')
plt.show()

#plt.show(sob)

plt.plot(bin_centers, hist, lw=2)
plt.show()


plt.show()

from PIL import Image
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

n = 10
l = 256
im = np.zeros((l, l))
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

mask = (im > im.mean()).astype(np.float)

mask += 0.1 * im

img = mask + 0.2*np.random.randn(*mask.shape)



binary_img = img > 0.5

plt.figure(figsize=(11,4))

plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.subplot(132)


plt.axvline(0.5, color='r', ls='--', lw=2)
plt.text(0.57, 0.8, 'histogram', fontsize=20, transform = plt.gca().transAxes)
plt.yticks([])
plt.subplot(133)
plt.imshow(binary_img, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
plt.show()im.

20 + 12
