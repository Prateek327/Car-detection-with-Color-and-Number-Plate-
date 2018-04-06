
#!/usr/bin/env python
import cv2
from colorsys import hsv_to_rgb
from math import sqrt, ceil
import matplotlib.colors
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import os
from scipy.cluster.vq import kmeans, whiten
import sys
import struct
from PIL import Image
import scipy
import scipy.misc
import scipy.cluster
from sklearn.cluster import KMeans


def crop(image_obj, coords, saved_location):
	cropped_image = image_obj.crop(coords)
	cropped_image.save(saved_location)


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
"""
def plot_colors(hist, cent):
    start = 0
    end = 0
    myRect = np.zeros((50, 300, 3), dtype="uint8")
    tmp = hist[0]
    tmpC = cent[0]
    for (percent, color) in zip(hist, cent):
        if(percent > tmp):
            tmp = percent
            tmpC = color
    end = start + (tmp * 300) # try to fit my rectangle 50*300 shape
    cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
                  tmpC.astype("uint8").tolist(), -1)
    start = end
    #rest will be black. Convert to black
    for (percent,color) in zip(hist, cent):
        end = start + (percent * 300)  # try to fit my rectangle 50*300 shape
        if(percent != tmp):
            color = [0, 0, 0]
            cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
                      color, -1) #draw in a rectangle
            start = end
    return myRect
"""
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar
	
	
def find_color(image,coords,saved_location):
	
	crop(image, ( 48.13239723443985,41.31784084439278,751.2476563453674,404.91936445236206 ), saved_location)
	img = cv2.imread(saved_location)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
	clt = KMeans(n_clusters=3) #cluster number
	clt.fit(img)
	hist = find_histogram(clt)
	print(clt.cluster_centers_)
	bar = plot_colors2(hist, clt.cluster_centers_)
	plt.axis("off")
	plt.imshow(bar)
	plt.show()
