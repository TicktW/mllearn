#!/usr/bin/env python
from PIL import Image as img
from matplotlib import pyplot as plt
from skimage import io, transform

##.pil_read_img
#wximg = img.open("/mnt/hgfs/project/weixin.jpg")
#wximg.show()
##.pil_read_img

##.skimg read
img2 = io.imread("/mnt/hgfs/project/weixin.jpg")
io.imshow(img2)
##.skimg read
