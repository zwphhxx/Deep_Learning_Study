import numpy as np
import matplotlib.pyplot as plt

img = np.zeros([200,200,3])
plt.imshow(img)
plt.show()

img = np.full([200,200,3],255)
plt.imshow(img)
plt.show()

img = plt.imread("./img/LenaRGB.bmp")
print("图像（H，W，C）：",img.shape)
plt.imshow(img)
plt.axis("off")
plt.show()