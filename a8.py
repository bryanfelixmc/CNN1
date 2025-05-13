import cv2
import numpy as np
import os
from pyautocad import Autocad, aDouble, APoint

acad = Autocad(create_if_not_exists=True)

def detect_shapes(image_path,
                  pixel_intensity_cutoff,
                  blur_k,
                  dp, minDist, param1, param2, minRadius, maxRadius,
                  canny_low, canny_high,
                  hl_threshold, hl_minLen, hl_maxGap):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Couldn’t load image at {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1) Binarize
    _, bw = cv2.threshold(
        gray,
        pixel_intensity_cutoff,   # sweep: e.g. 100→240
        255,
        cv2.THRESH_BINARY_INV
    )
    
    # 2) Blur
    blurred = cv2.GaussianBlur(bw, blur_k, 0)  # sweep: (3,3),(5,5),(7,7),(9,9)
    
    shapes = {'circles': [], 'lines': [], 'rectangles': []}
    
    # 3) HoughCircles
    detected_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,                 # 1.0,1.2,1.5,2.0
        minDist=minDist,       # 10,20,30,50
        param1=param1,         # 30,50,80,100
        param2=param2,         # 15,20,25,30
        minRadius=minRadius,   # 3,5,10
        maxRadius=maxRadius    # 100,200,300,400
    )
    if detected_circles is not None:
        for x, y, r in detected_circles[0]:
            shapes['circles'].append((float(x), float(y), float(r)))
    
    # 4) Canny + HoughLinesP
    edges = cv2.Canny(blurred,
                      canny_low,    # 10,20,30,50
                      canny_high,
                      apertureSize=3)
    detected_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=hl_threshold,   # 30,50,80,100
        minLineLength=hl_minLen,  # 5,10,20,40
        maxLineGap=hl_maxGap      # 2,5,10,20
    )
    if detected_lines is not None:
        for x1, y1, x2, y2 in detected_lines[:,0]:
            shapes['lines'].append((
                float(x1), float(y1),
                float(x2), float(y2)
            ))
    
    return shapes

# --------------------------------------------------------------------
# main sweep loop
image_folder = r"C:\Users\BMALPARTIDA\Documents\python\cad_draw_from_image\images"

# parameter grids
bin_threshes = range(100, 240, 20)                # 100,120,...,220
blur_kernels = [(3,3),(5,5),(7,7),(9,9)]
dp_vals       = [1.0, 1.2, 1.5, 2.0]
minDists      = [10, 20, 30, 50]
p1_vals       = [30, 50, 80, 100]
p2_vals       = [15, 20, 25, 30]
radii_min     = [3, 5, 10]
radii_max     = [100, 200, 300, 400]
canny_pairs   = [(30,100), (50,150)] #[(10,60), (20,80), (30,100), (50,150)]
hl_threshs    = [30, 50] #[30, 50, 80, 100]
hl_minLens    = [20, 40] #[5, 10, 20, 40]
hl_maxGaps    = [10, 20] #[2, 5, 10, 20]

d = 0
offset = 0

for thr in bin_threshes:
    for blur_k in blur_kernels:
        for dp in dp_vals:
            for md in minDists:
                for p1 in p1_vals:
                    for p2 in p2_vals:
                        for rmin in radii_min:
                            for rmax in radii_max:
                                for can_low, can_high in canny_pairs:
                                    for hl_t in hl_threshs:
                                        for hl_min in hl_minLens:
                                            for hl_gap in hl_maxGaps:
                                                d = 0
                                                
                                                #print(f"\n▶ sweep: thresh={thr}, blur={blur_k}, "
                                                #      f"dp={dp}, minDist={md}, p1={p1}, p2={p2}, "
                                                #      f"rmin={rmin}, rmax={rmax}, "
                                                #      f"canny=({can_low},{can_high}), "
                                                #      f"hl=({hl_t},{hl_min},{hl_gap})")
                                                
                                                for img_file in os.listdir(image_folder):
                                                    path = os.path.join(image_folder, img_file)
                                                    res = detect_shapes(
                                                        path,
                                                        thr, blur_k,
                                                        dp, md, p1, p2, rmin, rmax,
                                                        can_low, can_high,
                                                        hl_t, hl_min, hl_gap
                                                    )
                                                    print(f"  {img_file}: circles={len(res['circles'])}, "
                                                          f"lines={len(res['lines'])}")
                                                    
                                                    # draw into AutoCAD
                                                    for cx, cy, cr in res['circles']:
                                                        c = acad.model.AddCircle(
                                                            APoint(cx + d, cy + offset),
                                                            cr
                                                        )
                                                        c.Color = 1
                                                    for x1,y1,x2,y2 in res['lines']:
                                                        l = acad.model.AddLine(
                                                            APoint(x1 + d, y1 + offset),
                                                            APoint(x2 + d, y2 + offset)
                                                        )
                                                        l.Color = 3
                                                    
                                                    texto1=f"sweep: thresh={thr}, blur={blur_k}, dp={dp}, minDist={md}, p1={p1}, p2={p2}, rmin={rmin}, rmax={rmax}, canny=({can_low},{can_high}), hl=({hl_t},{hl_min},{hl_gap})"
                                                    acad.model.AddMText(APoint(100+d, offset, 0), 100, texto1)
                                                    d += 500
                                                offset -= 400
