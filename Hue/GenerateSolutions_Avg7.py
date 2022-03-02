import cairo
from math import pi
import math
from PIL import Image
import numpy as np
import map_elites.cvt as cvt_map_elites
import map_elites.common as cm_map_elites
from sklearn.neighbors import KDTree

def cosine_similarity(n1, n2):
    cos_sim = np.dot(n1,n2) / (np.linalg.norm(n1)*np.linalg.norm(n2))
    return cos_sim

def draw(geno):

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)

    ctx.set_source_rgba(geno[0], geno[1], geno[2], 1)
    ctx.paint()

    return surface


def fit(geno, arch):

    img = draw(geno)
    imgP = Image.frombuffer("RGBA",(img.get_width(),img.get_height() ),img.get_data(),"raw","RGBA",0,1)
    img_HSV = imgP.convert('HSV')
    img_HSV_arr = np.array(img_HSV)

    s_arr = img_HSV_arr[:,:,1]
    v_arr = img_HSV_arr[:,:,2]

    s = np.max(s_arr) / 255
    v = np.max(v_arr) / 255

    kdt = cvt_map_elites.getKDT(1000, 2)
    test = kdt.query([np.array([s, v])], k=8)[1][0]
    current_niche = kdt.data[test[0]]
    n = cm_map_elites.make_hashable(current_niche)

    if n in arch:

        np.delete(test, 0)
        dist_hold = []
        for j in range(len(test)):

            nei_niche = kdt.data[test[j]]
            nei = cm_map_elites.make_hashable(nei_niche)
            if nei in arch:
                img_a = draw(arch[nei].x)
                img = draw(geno)
                imgNEW = Image.frombuffer("RGBA",( img_a.get_width(),img_a.get_height() ),img_a.get_data(),"raw","RGBA",0,1)
                imgOLD = Image.frombuffer("RGBA",( img.get_width(),img.get_height() ),img.get_data(),"raw","RGBA",0,1)

                imgN_HSV = imgNEW.convert('HSV')
                imgN_HSV_arr = np.array(imgN_HSV)
                h_new = np.max(imgN_HSV_arr[:,:,0])*360 / 255

                imgO_HSV = imgOLD.convert('HSV')
                imgO_HSV_arr = np.array(imgO_HSV)
                h_old = np.max(imgO_HSV_arr[:,:,0])*360 / 255
                #dist = cosine_similarity(h_new, h_old)
                #dist = abs(h_new - h_old) /360
                hueDistance = min(abs(h_new-h_old), 360-abs(h_new-h_old)) / 180
                dist_hold.append(hueDistance)
            else:
                dist_hold.append(0)

        img = draw(geno)
        imgP = Image.frombuffer("RGBA",(img.get_width(),img.get_height() ),img.get_data(),"raw","RGBA",0,1)
        img_HSV = imgP.convert('HSV')
        img_HSV_arr = np.array(img_HSV)

        s_arr = img_HSV_arr[:,:,1]
        v_arr = img_HSV_arr[:,:,2]

        s = np.max(s_arr) / 255
        v = np.max(v_arr) / 255

        f = -(sum(dist_hold) / len(dist_hold))
        
        print("fitness : {}".format(f))
        return f, np.array([s , v ])

    else:

        img = draw(geno)
        imgP = Image.frombuffer("RGBA",(img.get_width(),img.get_height() ),img.get_data(),"raw","RGBA",0,1)
        img_HSV = imgP.convert('HSV')
        img_HSV_arr = np.array(img_HSV)

        s_arr = img_HSV_arr[:,:,1]
        v_arr = img_HSV_arr[:,:,2]

        s = np.max(s_arr) / 255
        v = np.max(v_arr) / 255
        f = - 1
        print("Was empty")
        return f, np.array([s , v ])

    
px = cm_map_elites.default_params.copy()

if __name__ == '__main__':
    archive = cvt_map_elites.compute(2, 3, fit, n_niches=1000, max_evals=100000, log_file=open('cvt.dat', 'w'), params=px)				