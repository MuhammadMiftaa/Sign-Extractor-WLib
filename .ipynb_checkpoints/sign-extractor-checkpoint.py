

''' 
Program Berbasis Python untuk Mengekstrak atau menyeleksi tanda tangan
'''
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np
from math import log10, sqrt 


def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def PSNRread(): 
     original = cv2.imread("./inputs/sign2.jpg") 
     compressed = cv2.imread("./outputs/output.png", 1) 
     value = PSNR(original, compressed) 
     print(f"PSNR value is {value} dB") 

def sauvola_threshold(image, window_size=25, k=0.2, r=128):
    # Matriks kernel untuk operasi konvolusi
    kernel = np.ones((window_size, window_size), np.uint8)

    # Konvolusi gambar dengan kernel yang sama
    mean = cv2.filter2D(image, -1, kernel / (window_size ** 2))

    # Hitung deviasi standar menggunakan operasi konvolusi dengan kernel yang sama
    dev = np.sqrt(cv2.filter2D(image ** 2, -1, kernel / (window_size ** 2)) - mean ** 2)

    # Hitung nilai threshold Sauvola
    threshold = mean * (1 + k * (dev / r - 1))

    # Binarisasi gambar
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
    return binary_image

'''
Import gambar yang ingin diproses (.jpg, .jpeg, .png)
Lalu citra tanda tangan dikonversi ke citra biner => 127 : nilai ambang batas; 255 : nilai maksimum
Menggunakan transformasi biner sederhana
'''
img = cv2.imread('./inputs/sign2.jpg', 0)
# img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] 
img = sauvola_threshold(img)
# cv2.imwrite("./alur/1.png", img)
cv2.imwrite("./alur/1.png", img)


'''
Parameter konstan sebagai nilai-nilai ambang batas (threeshold) yang berfungsi untuk menghilangkan 
nilai(outlier) dan juga noise disekitar tanda tangan yang terlalu kecil dan terlalu besar.
'''
parameter_konstan_1 = 84  
parameter_konstan_2 = 250 
parameter_konstan_3 = 100 
parameter_konstan_4 = 37  


'''
SEGMENTASI CITRA => Menganalis atau menyeleksi titik-titik yang terhubung dan membentuk tanda tangan
Variabel blob untuk menyimpan kumpulan biner dari proses konversi citra binary. Operasi ini menghasilkan sebuah citra 
yang menunjukkan piksel mana yang memiliki intensitas di atas rata-rata dan mana yang di bawah rata-rata.
'''
blobs = img > img.mean()
'''
blobs_labels adalah citra mengandung informasi tentang komponen-komponen terhubung yang ada dalam citra biner blobs, 
yang dapat digunakan untuk analisis lebih lanjut atau untuk visualisasi komponen-komponen tersebut dalam konteks gambar asli.
'''
blobs_labels = measure.label(blobs, background=1)
cv2.imwrite("./alur/2.png", blobs_labels)

''' Inisialisasi variabel the_biggest_component, total_area, counter, dan average untuk menyimpan statistik komponen-komponen terhubung. '''
komponen_terbesar = 0
total_area = 0
counter = 0
rata_rata = 0.0
'''
Pada setiap iterasi, region.area diperiksa. Jika luas komponen > 10 maka tambahkan luasnya ke total_area juga counter+1 dan 
rata_rata membantu dalam menentukan ambang batas yang sesuai untuk membersihkan gambar dari noise.
'''
for region in regionprops(blobs_labels):
    if (region.area > 10):
        total_area = total_area + region.area
        counter = counter + 1
rata_rata = (total_area/counter)

''' 
small_threshold digunakan sebagai threshold untuk menghapus piksel yang terhubung dengan outlier yang lebih kecil 
big_threshold digunakan sebagai threshold untuk menghapus piksel yang terhubung dengan outlier yang lebih besar 
'''
small_threshold = ((rata_rata/parameter_konstan_1)*parameter_konstan_2)+parameter_konstan_3
print("Threshold Minimum: ", small_threshold)

big_threshold = small_threshold*parameter_konstan_4
print("Threshold Maximum: ", big_threshold)

''' Menghapus pixels yang lebih kecil daripada threshold small_threshold'''
pre_version = morphology.remove_small_objects(blobs_labels, small_threshold)
cv2.imwrite('./alur/3.png', pre_version)

''' Menghapus pixels yang lebih besar daripada threshold big_threshold dan menghapus pixels terhubung lain yang mengganggu '''
component_sizes = np.bincount(pre_version.ravel())
too_small = component_sizes > (big_threshold)
too_small_mask = too_small[pre_version]
pre_version[too_small_mask] = 0

''' Menyimpan citra binary versi awal'''
cv2.imwrite('./alur/4.png', pre_version)

''' Mengkonversi binary dari metode Otsu dan Invers'''
img = cv2.imread('./alur/4.png', 0)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imwrite("./alur/5.png", img)
cv2.imwrite("./outputs/output.png", img)
PSNRread()


