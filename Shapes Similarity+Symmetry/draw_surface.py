import cairo

def draw(geno):
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 300, 300)
    ctx = cairo.Context(surface)
    
    cr = ctx
    cr.scale(150, 150)
    
    cr.set_line_width(geno[54]*geno[55])
    cr.set_source_rgba(geno[0], geno[1], geno[2], geno[3])
    cr.move_to(geno[4]*2, geno[5]*2)
    cr.curve_to(geno[6]*2, geno[7]*2, geno[8]*2, geno[9]*2, geno[10]*2, geno[11]*2)
    cr.curve_to(geno[12]*2, geno[13]*2, geno[14]*2, geno[15]*2, geno[16]*2, geno[17]*2)
    cr.move_to(geno[18]*2, geno[19]*2)
    cr.curve_to(geno[20]*2, geno[21]*2, geno[22]*2, geno[23]*2, geno[24]*2, geno[25]*2)
    cr.curve_to(geno[26]*2, geno[27]*2, geno[28]*2, geno[29]*2, geno[30]*2, geno[31]*2)
    cr.move_to(geno[32]*2, geno[33]*2)
    cr.curve_to(geno[34]*2, geno[35]*2, geno[36]*2, geno[37]*2, geno[38]*2, geno[39]*2)
    cr.curve_to(geno[40]*2, geno[41]*2, geno[42]*2, geno[43]*2, geno[44]*2, geno[45]*2)
    cr.move_to(geno[46]*2, geno[47]*2)
    cr.curve_to(geno[48]*2, geno[49]*2, geno[50]*2, geno[51]*2, geno[52]*2, geno[53]*2)
    cr.stroke()
        
    return surface