from multiprocessing import Pool
import cv2
import glob
import numpy as np

def blur(img_path, outfolder=""):
    img = cv2.imread(img_path)
    blurred = cv2.GaussianBlur(img, (7,7), 5)
    img_name = img_path.split('/')[-1]
    cv2.imwrite('data/sah_blur/{}'.format(img_name), blurred)

def contrast(img_path):
    img = Image.open(img_path)
    res = adjust_contrast(img, 2)

    img_name = img_path.split('/')[-1]
    res.save('data/sdh_contrast/{}'.format(img_name))

    
if __name__ == '__main__':
    # test blur
    pool = Pool(4)
    paths = glob.glob("data/sah/*")
    pool.map(blur, paths)
    
    # test contrast
    paths = glob.glob("data/sdh/*")
    pool.map(contrast,paths)    