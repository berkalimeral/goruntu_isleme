import cv2
import matplotlib.pyplot as plt

imgpath = 'image.jpg'
img = cv2.imread(imgpath)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.imread(imgpath, 0)

colored_negative = abs(255-img)
gray_negative = abs(255-gray)

imgs = [img, gray, colored_negative, gray_negative]
title = ['coloured', 'negative']


plt.subplot(2, 2, 1)
plt.title(title[0])
plt.imshow(imgs[0])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.title(title[1])
plt.imshow(imgs[2])
plt.xticks([])
plt.yticks([])


plt.show()