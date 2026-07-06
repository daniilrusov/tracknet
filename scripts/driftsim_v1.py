#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import numpy as np
from tqdm import tqdm

xmin = -1500
ymin = -1500
umin = -1500
vmin = -1500
uvangle = 45*3.14159/180. # deg to rad
width = 10

def LineExtrapToZ(x0, y0, theta, phi, Z):
    perp = Z * math.tan(theta)
    x = x0 + perp*math.cos(phi)
    y = y0 + perp*math.sin(phi)
    z = float(Z)
    return (x,y,z)

def GetWireDrift(x, xmin):
    wirenum = round((x-xmin)/width)
    dR = (x-xmin)-wirenum*width
    lr = 1
    if dR<0:
        dR = -dR
        lr = -1
    return (wirenum, dR, lr)

def GetWirePos(wireid):
    station = round(wireid/1000)
    wirenum = wireid%1000

    zplane = np.linspace(0, 240, 9) # mm
    pos = wirenum*width
    
    if (station == 1) or (station == 5): # X
        x0 = xmin + pos
        y0 = ymin
        z0 = zplane[station-1]
        xyuv=1
    if (station == 2) or (station == 6): # Y
        x0 = xmin 
        y0 = ymin + pos
        z0 = zplane[station-1]
        xyuv = 2
    if (station == 3) or (station == 7): # U
        x0 = xmin + pos/math.cos(uvangle)
        y0 = ymin + pos//math.sin(uvangle)
        z0 = zplane[station-1]
        xyuv = 3
    if (station == 4) or (station == 8): # V
        x0 = xmin + pos/math.sin(uvangle)
        y0 = ymin + pos/math.cos(uvangle)
        z0 = zplane[station-1]
        xyuv = 4

    return (x0,y0,station,z0,xyuv)


def GetWire(x,y,station):
    # wireid = station*1000+tubeid
    # tubeid [0,150]
    # 8 stations [1,8]: x1 y1 u1 v1 x2 y2 u2 v2

    uvangle = 45*3.14159/180. # deg to rad
    
    if (station == 1) or (station == 5): # X
        pos = x
        posmin = xmin
    if (station == 2) or (station == 6): # Y
        pos = y
        posmin = ymin
    if (station == 3) or (station == 7): # U
        pos = math.sqrt(x*x+y*y)*math.cos(math.atan2(x,y)-uvangle)
        posmin = math.sqrt(xmin*xmin+ymin*ymin)
    if (station == 4) or (station == 8): # V
        pos = math.sqrt(x*x+y*y)*math.sin(math.atan2(x,y)-uvangle)
        posmin = math.sqrt(xmin*xmin+ymin*ymin)
        
    wirenum, dR, lr = GetWireDrift(pos, posmin)
    wireid = station*1000 + wirenum
    
    return (wireid,dR,lr)



if __name__ == '__main__':
    nevents = int(sys.argv[1])
    
    eff = 1.#0.98  # detector efficiency

    # dimensions xy: 1500 x 1500
    # tube diameter 10 mm
    # 8 planes: x1 y1 u1 v1 x2 y2 u2 v2
    zplane = np.linspace(0, 240, 8) # mm

    with open('output.tsv', 'w') as f:
        for evt in tqdm(range(0, nevents)):
            pi = 3.14156

            vtxx = random.uniform(-700, 700)
            vtxy = random.uniform(-700, 700)
            
            ntrk = int(random.uniform(1,10))

            for trk in range(0, ntrk):
                pt = random.uniform(100,1000) # MeV/c
                phi = random.uniform(0, 2*pi)
                theta = math.acos(random.uniform(0,1)) #formard tracks

                charge = 0

                #while charge == 0:
                #    charge = random.randint(-1,1)

                station = 1
                for Z in zplane:
                    x,y,z = LineExtrapToZ(vtxx, vtxy, theta, phi, Z)

                  #  if (x,y,z) == (0,0,0):
                  #      continue
                    if math.fabs(x) >= 750 or math.fabs(y) >= 750 :
                        continue

                    
                    wireid,dr,lr = GetWire (x,y,station)
                    x0,y0,station,z0,xyuv = GetWirePos(wireid)

 #                   if random.uniform(0,1) < eff:
                    f.write("%d\t%d\t%f\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%d\n" % (evt,wireid,dr,lr,station,trk,x,y,x0,y0,z0) )
   #                 print(evt,wireid,dr,lr,station,trk,x,y,x0,y0,z0)
                    station = station + 1

#    # add noise hits
#            nhit = int(random.uniform(ntrk * ntrk * 35/2, ntrk * ntrk * 35/1)) # up to 100 noise hits
#            for ihit in range(0, nhit):
#                sta = int(random.uniform(0,35))
#                R = radii[sta]
#                phi = random.uniform(0, 2*pi)
#                z = random.uniform(-2386, 2386)
#                x = R*math.cos(phi)
#                y = R*math.sin(phi)
#                f.write("%d\t%f\t%f\t%f\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (evt,x,y,z,sta,-1,0,0,0,0,0,0) )
        f.close()
