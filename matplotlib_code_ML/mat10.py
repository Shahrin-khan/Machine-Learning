import numpy as np
import matplotlib.pyplot as plt
img1=plt.imread('D:\\DIP\\Dataset\\misc\\4.1.06.tiff')
print(np.median(img1))
print(np.average(img1))
print(np.mean(img1))
print(np.std(img1))
print(np.var(img1))
