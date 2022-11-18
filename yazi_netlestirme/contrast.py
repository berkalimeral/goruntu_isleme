import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg")
image = cv2.resize(image, (500, 600))
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=5)
final_img = clahe.apply(image_bw) + 30
_, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
plt.title('Orjinal Resim'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(final_img, cmap='gray')
plt.title('CLAHE image'), plt.xticks([]), plt.yticks([])

plt.show()