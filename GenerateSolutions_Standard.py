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
    #ctx.set_operator(cairo.OPERATOR_ADD)
    #ctx.save()
    #ctx.set_source_rgb(0, 0, 0)
    #ctx.paint()
    #ctx.restore()
    
    
    #ctx.save()
    #ctx.translate(50,50)
    ctx.set_source_rgba(geno[0], geno[1], geno[2], 1)
    #ctx.arc(0, 0, 40, 0, 2*math.pi)
    #ctx.fill()
    ctx.paint()
    #ctx.restore()

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

    f = 0
        
    #print("fitness : {}".format(f))

    return f, np.array([s , v ])

    
px = cm_map_elites.default_params.copy()

if __name__ == '__main__':
    archive = cvt_map_elites.compute(2, 3, fit, n_niches=1000, max_evals=100000, log_file=open('cvt.dat', 'w'), params=px)				