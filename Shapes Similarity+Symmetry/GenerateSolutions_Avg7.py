import cairo
from math import pi
import math
from PIL import Image
import numpy as np
import map_elites.cvt as cvt_map_elites
import map_elites.common as cm_map_elites
from sklearn.neighbors import KDTree
import cv2
from scipy.spatial import distance


t = math.pi/180.0


def polar_to_cart(theta, dist):
    
    x = 1 + dist * math.cos(theta)
    y = 1 + dist * math.sin(theta)
    
    return x,y

def remap(old_val, old_min, old_max, new_min, new_max):
    return (new_max - new_min)*(old_val - old_min) / (old_max - old_min) + new_min

def draw(geno):
    
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 100)
    ctx = cairo.Context(surface)

    ctx.scale(50, 50)
    # Paint the background
    ctx.set_source_rgb(0, 0 , 0)
    ctx.paint()

    r1 = remap(geno[8], 0, 1,0.1, 1)
    r2 = remap(geno[9], 0, 1,0.1, 1)
    r3 = remap(geno[10], 0, 1,0.1, 1)
    r4 = remap(geno[11], 0, 1,0.1, 1)
    r5 = remap(geno[12], 0, 1,0.1, 1)
    r6 = remap(geno[13], 0, 1,0.1, 1)
    r7 = remap(geno[14], 0, 1,0.1, 1)
    r8 = remap(geno[15], 0, 1,0.1, 1)

    # Draw the image
    firstx, firsty = polar_to_cart((0 + geno[0])*45*t, r1)
    secondx, secondy = polar_to_cart((1 + geno[1])*45*t, r2)
    thirdx, thirdy = polar_to_cart((2 + geno[2])*45*t, r3)
    forthx, forthy = polar_to_cart((3 + geno[3])*45*t, r4)
    fifthx, fifthy = polar_to_cart((4 + geno[4])*45*t, r5)
    sixthx, sixthy = polar_to_cart((5 + geno[5])*45*t, r6)
    seventhx, seventhy = polar_to_cart((6 + geno[6])*45*t, r7)
    eigthx, eigthy = polar_to_cart((7 + geno[7])*45*t, r8)
    ctx.move_to(firstx, firsty)

    ctx.line_to(secondx, secondy)
    ctx.line_to(thirdx, thirdy)
    ctx.line_to(forthx, forthy)
    ctx.line_to(fifthx, fifthy)
    ctx.line_to(sixthx, sixthy)
    ctx.line_to(seventhx, seventhy)
    ctx.line_to(eigthx, eigthy)
    
    ctx.close_path()
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill_preserve()
    

    return surface


def fit(geno, arch):

    img = draw(geno)
    imgP1 = Image.frombuffer("RGBA",( img.get_width(),img.get_height() ),img.get_data(),"raw","RGBA",0,1)

    imgrey = imgP1.convert('L')
    im = np.array(imgrey)
    ret, thresh = cv2.threshold(im,62,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = cv2.boundingRect(contours[0])


    kdt = cvt_map_elites.getKDT(1000, 2)
    test = kdt.query([np.array([w/100, h/100])], k=8)[1][0]
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

                imgN_arr = np.array(imgNEW)
                imgN= imgN_arr[:,:,0].flatten()
                imgO_arr = np.array(imgOLD)
                imgO= imgO_arr[:,:,0].flatten()

                #dist = levenshtein(imgN, imgO)
                dist = distance.hamming(imgN,imgO)
                
                dist_hold.append(dist)
            else:
                dist_hold.append(0)

        img = draw(geno)
        imgP = Image.frombuffer("RGBA",(img.get_width(),img.get_height() ),img.get_data(),"raw","RGBA",0,1)
        imgrey = imgP.convert('L')
        im = np.array(imgrey)
        ret, thresh = cv2.threshold(im,62,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        x,y,w,h = cv2.boundingRect(contours[0])

        #f = -max(dist_hold)
        similarity = -(sum(dist_hold) / len(dist_hold)) / 100*100*0.79
        #angles = abs(geno[0] - geno[4]) + abs(geno[1] - geno[5]) + abs(geno[2] - geno[6]) + abs(geno[3] - geno[7])
        #radius = abs(geno[8] - geno[12]) + abs(geno[9] - geno[13]) + abs(geno[10] - geno[14])  + abs(geno[11] - geno[15])
        #simmetry = - (radius + angles) / (2*3.6)

        #f = simmetry + similarity
        r1 = remap(geno[8], 0, 1,0.1, 1)
        r2 = remap(geno[9], 0, 1,0.1, 1)
        r3 = remap(geno[10], 0, 1,0.1, 1)
        r4 = remap(geno[11], 0, 1,0.1, 1)
        r5 = remap(geno[12], 0, 1,0.1, 1)
        r6 = remap(geno[13], 0, 1,0.1, 1)
        r7 = remap(geno[14], 0, 1,0.1, 1)
        r8 = remap(geno[15], 0, 1,0.1, 1)

        x = abs(r1*math.cos(geno[0]*45*t) - r5*math.cos(geno[4]*45*t)) + abs(r2*math.cos(geno[1]*45*t) - r6*math.cos(geno[5]*45*t)) + abs(r3*math.cos(geno[2]*45*t) - r7*math.cos(geno[6]*45*t)) + abs(r4*math.cos(geno[3]*45*t) - r8*math.cos(geno[7]*45*t))
        y = abs(r1*math.sin(geno[0]*45*t) - r5*math.sin(geno[4]*45*t)) + abs(r2*math.sin(geno[1]*45*t) - r6*math.sin(geno[5]*45*t)) + abs(r3*math.sin(geno[2]*45*t) - r7*math.sin(geno[6]*45*t)) + abs(r4*math.sin(geno[3]*45*t) - r8*math.sin(geno[7]*45*t))

        f = -(x + y) + similarity
        #print("fitness : {}".format(f))
        return f, np.array([w/100 , h/100 ])

    else:

        img = draw(geno)
        imgP = Image.frombuffer("RGBA",(img.get_width(),img.get_height() ),img.get_data(),"raw","RGBA",0,1)


        imgrey = imgP1.convert('L')
        im = np.array(imgrey)
        ret, thresh = cv2.threshold(im,62,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        x,y,w,h = cv2.boundingRect(contours[0])

        f = - 1
        #print("Was empty")
        return f, np.array([w/100 , h/100 ])


    
px = cm_map_elites.default_params.copy()

if __name__ == '__main__':
    archive = cvt_map_elites.compute(2, 16, fit, n_niches=1000, max_evals=100000, log_file=open('cvt.dat', 'w'), params=px)				