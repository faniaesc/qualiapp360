#IMPORTS NEEDED
from tkinter import * 
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import tkinter.font as tkf
import subprocess
from cv2 import *
import cv2 as cv
from numpy import * 
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from moviepy.editor import *
import sys
import argparse
import msvcrt
from skimage.metrics import structural_similarity as ssim
from zipfile import ZipFile
import wget
import os
import glob
import re
import xml.etree.ElementTree as ET
import csv
import math
import time


# specifying the zip file name
file_name       = 'voronoiMetrics.zip'
thirdParty_zip  = 'voronoiMetricsThirdParty.zip'
project_name    = 'voronoiVMAF/'
video_folder    = 'videos/'
vpatch          = 'ExeAndDlls/OmniVideoQuality.exe'
vmaf_model      = 'vmaf_rb_v0.6.3/vmaf_rb_v0.6.3.pkl'

       
# METRICS

# PSNR
def getPSNRdata(I1, I2):
    
    ix_iy = absdiff(I1, I2)                  # absolute difference between the two frames
    ix_iy = float32(ix_iy)                   # cannot make a square on 8 bits
    ix_iy_2 = ix_iy ** 2                     # (Ix - Iy)^2
    summationPSNR = ix_iy_2.sum()            # Sum of (Ix - Iy)^2

    if(summationPSNR <= 1e-10):              # check if the value is too small
        return 0                             # then return zero
    else:
        shape = I1.shape                     # shape of an array is the number of elements in each dimension
        cij = shape[0] * shape[1] * shape[2] # c * i * j
        mse = summationPSNR / (cij)          # sum of (Ix - Iy)**2 / c * i * j
        maxI = 255                           # 1 byte per pixel per channel = 255
        psnr = 10.0 * log10((maxI ** 2) / mse) 
        return psnr

# MSSIM
def getMSSIMdata(I1, I2):
   
    K1 = 0.01
    K2 = 0.03
    L = 255                                  # Bitdepth of an image
    C1 = (K1*L)**2                           # Const value of C1
    C2 = (K2*L)**2                           # Const value of C2
  
    i1 = float32(I1)                         # cannot calculate on one byte large values
    i2 = float32(I2)
    i1_2 = i1 ** 2                           # I1^2
    i2_2 = i2 ** 2                           # I2^2
    i1_i2 = i1 * i2                          # I1 * I2
    
    mu_x = cv.GaussianBlur(i1, (11, 11), 1.5) #GaussianBlur(src, size of the kernel, sigmaX)
    mu_y = cv.GaussianBlur(i2, (11, 11), 1.5)

    mu_x_2 = mu_x * mu_x
    mu_y_2 = mu_y * mu_y
    mu_x_mu_y = mu_x * mu_y

    sigma_x_2 = cv.GaussianBlur(i1_2, (11, 11), 1.5)
    sigma_y_2 = cv.GaussianBlur(i2_2, (11, 11), 1.5)
    sigma_x_y = cv.GaussianBlur(i1_i2, (11, 11), 1.5)

    sigma_x_2 -= mu_x_2
    sigma_y_2 -= mu_y_2
    sigma_x_y -= mu_x_mu_y

    #SSIM(x,y)
    step1 = 2 * mu_x_mu_y + C1
    step2 = 2 * sigma_x_y + C2
    step3 = step1 * step2                    # t3 = ((2*mu_x_mu_y + C1).*(2*sigma_x2 + C2))
    step4 = mu_x_2 + mu_y_2 + C1
    step5 = sigma_x_2 + sigma_y_2 + C2
    step6 = step4 * step5                    # t1 =((mu_x_2 + mu_y_2 + C1).*(sigma_x_2 + sigma_y_2 + C2))

    ssim_map = cv.divide(step3, step6)    # ssim_map =  t3./t1;
    mssim = cv.mean(ssim_map)       # mssim = average of ssim map
    print(mssim)
    return mssim

# SSIM
def getSSIMdata(I1, I2):

    # Convert the images to grayscale
    gray1 = cv.cvtColor(I1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(I2, cv.COLOR_BGR2GRAY)

    # Check for same size and ratio and report accordingly
    ho, wo, _ = I1.shape
    hc, wc, _ = I2.shape
    ratio_orig = ho/wo
    ratio_comp = hc/wc

    if round(ratio_orig, 2) == round(ratio_comp, 2):
        ssim_value = compare(gray1, gray2)
    return ssim_value

#Needed for SSIM
def compare(imageA, imageB):
        # Calculate the SSIM
        s = ssim(imageA, imageB)
        # Return the SSIM. The higher the value, the more "similar" the two images are.
        return s

def getWSPSNRdata(self, I1, I2, i, weightMap):
    
    total_frames = 0
    total = 0
    duration = 0

    weightedDiff = np.ones((self.global_width, self.global_height), dtype=np.float64)
    weightedDiffAux = np.ones((self.global_width, self.global_height), dtype=np.float64)
    diffMap = np.ones((self.global_width, self.global_height), dtype=np.float64)
    srcImg = np.ones((self.global_width, self.global_height), dtype=np.float64)
    dstImg = np.ones((self.global_width, self.global_height), dtype=np.float64)

    srcImg = ReadWS(self, I1)
    dstImg = ReadWS(self, I2)

    while(total_frames < i):
        self.frame_actual += 1
        self.progressBar['value'] = self.frame_actual 
        self.root.update_idletasks()

        srcImg = ReadWS(self, srcImg)
        dstImg = ReadWS(self, dstImg)

        start = time.time()

        cv.absdiff(srcImg, dstImg , diffMap)
        cv.pow(diffMap, 2, diffMap)

        weightedDiffAux = np.multiply(diffMap, weightMap)
        weightedDiff = np.multiply(weightedDiffAux, 100000)

        WMSE = sum(weightedDiff[i]) / sum(weightMap[i]) / 100000
        WSPSNR = 10 * log10(255 * 255 / (WMSE + sys.float_info.epsilon))

        end = time.time()
        t = end - start
        
        print("Frame: {} WS-PSNR score: {} Time: {}".format(total_frames, WSPSNR, t, end=" "))
        print()

        total += WSPSNR
        duration += t
        total_frames += 1

    return total, duration
    
def ReadWS(self, frame):        
    if frame is not None:
        aux_frame = cv.cvtColor(frame, cv.COLOR_BGRA2YUV_I420)
        cv.rectangle(aux_frame, (0, 0), (self.global_width, self.global_height), (255,0,0))
    return frame
    
def VideoCaptureYUV(self, filename, width, height):
    try:
        self.height = height
        self.width = width
        self.frame_len = self.width * self.height * 3 / 2
        # Open '*.yuv' as a binary file.
        self.f = open(filename, 'rb')
        self.shape = (int(self.height*1.5), self.width)
    except:
        messagebox.showerror(message="An error has occurred while choosing the file.", title="Error")

# S-PSNR
def getSPSNRdata(self, sph_points_x, sph_points_y, n_sphere, src, dst):
    sum = 0
    i = 0
    cart = []
    w = self.global_width
    h = self.global_height

    while(i < n_sphere):
        
        self.progressBar.config(maximum= self.n_sphere)
        self.progressBar['value'] = i 
        self.root.update_idletasks()

        self.frame_actual += 1
        self.progressBar['value'] = self.frame_actual 
        self.root.update_idletasks()

        cart = convSphToCart(self, sph_points_x[i], sph_points_y[i], cart)
        x, y = convCartToRect(self, w, h, cart)

        x -= 0.5
        y -= 0.5

        val1 = filter(self, src, x, y, w, h, w)
        val2 = filter(self, dst, x, y, w, h, w)
        diff = val1 - val2
        diff = abs(diff)
        sum += diff * diff

        i += 1

    sum /= n_sphere

    if sum == 0:
        sum = 100
    else:
        sum = 10*log10(255*255 / sum)

    return sum

def filter(self, src, x, y, w, h, s_src):
    width_LB = 1
    width_HB = w - 2
    height_LB = 1
    height_HB = h - 2

    if((y <= height_LB) | (y >= height_HB) | (x <= width_LB) | (x >= width_HB)):

        x = min(x, w-1)
        x = max(x, 0, w-1)      
        y = min(y, h-1)
        y = max(y, 0, h-1)      

        filter_result = linear_interp(self, src, x, y, s_src)

    else:
        filter_result = cubic_interp(self, src, x, y, s_src)

    return filter_result

def cubic_interp(self, src, x, y, s_src):

    x_idx = []
    y_idx = []
    val = []
    i = 0

    x_idx[0] = (int(x - 1))
    x_idx[1] = x_idx[0] + 1
    x_idx[2] = x_idx[1] + 1
    x_idx[3] = x_idx[2] + 1
    y_idx[0] = (int(y - 1))
    y_idx[1] = y_idx[0] + 1
    y_idx[2] = y_idx[1] + 1
    y_idx[3] = y_idx[2] + 1

    
    while(i < 4):
        aZ = src + y_idx[i] * s_src + x_idx[0]
        aZero = src + y_idx[i] * s_src + x_idx[1]
        a1 = src + y_idx[i] * s_src + x_idx[2]
        a2 = src + y_idx[i] * s_src + x_idx[3]
        t = x - x_idx[1]
        val[i] = cubicinterpolation(aZ, aZero, a1, a2, t)
        i += 1
    
    filter_res = cubicinterpolation(val[0], val[1], val[3], y- y_idx[1] + 0.5)
    return filter_res

def cubicinterpolation(self, aZ, aZero, a1, a2, t):

    cZ = 2 * aZero
    cZero = -aZ + a1
    c1 = 2 * aZ - 5 * aZero + 4 * a1 - a2
    c2 = -aZ + 3 * aZero - 3 * a1 + a2

    t2 = t * t
    t3 = t2 * t

    v = 0.5 * (cZ + cZero*t + c1*t2 + c2*t3)

    x = min(v, 255)
    x = max(x, 0, 255)      
    y = min(v, 255)
    y = max(y, 0, 255)   

    return


def linear_interp(self, src, x, y, s_src):

    x_lo = x[0]
    y_lo = y[0]
    x_hi = ceil(x)
    y_hi = ceil(y)

    try:
        a1 = x_lo + y_lo*s_src
        a1 = src[a1[0]]
    except:
        a1 = 0

    try:
        b1 = x_hi + y_lo*s_src
        b1 = src[b1[0]]
    except:
        b1 = 0
        
    k1 = x - x_lo

    #interpolation
    val1 = a1 * (1 - k1) + b1 * k1

    try:
        a2 = x_lo + y_hi*s_src
        a2 = src[a2[0]]
    except:
        a2 = 0

    try:
        b2 = x_hi + y_hi*s_src
        b2 = src[b2[0]]
    except:
        b2 = 0

    k2 = x - x_lo

    #interpolation
    val2 = a2 * (1 - k2) + b2 * k2

    k3 = y - y_lo

    #interpolation
    val = val1 * (1 - k3) + val2 * k3

    return val

def convCartToRect(self, w, h, cart):

    x = cart[0]
    y = cart[1]
    z = cart[2]

    phi = math.acos(y)
    theta = math.atan2(x,z)

    x_out = w * (0.5 + theta / math.pi)
    y_out = h * (phi / math.pi)

    return x_out, y_out

def convSphToCart(self, sph_x, sph_y, cart):   

    sph_lat = sph_x
    sph_lon = sph_y
    lat = (float(sph_lat) * math.pi / 180)
    lon = (float(sph_lon) * math.pi / 180)

    cart.append(sin(lon) * cos(lat))
    cart.append(sin(lat))
    cart.append(-cos(lon) * cos(lat))

    return cart

class QualiApp360:
    def __init__(self):

        self.root = Tk()
        self.root.title("QualiApp360")
        self.root.geometry("1000x600")
        self.root.configure(bg="white")
        myFont = tkf.Font(family="Nyata FTR", size=14)
        myFontLittle = tkf.Font(family="Nyata FTR", size=13)
        self.style = ttk.Style()
        self.style.theme_use('default')
        
        #Checks for Start Evaluation
        self.check1 = False
        self.check2 = False
        self.check3 = False
        self.spherical_txt = ""

        #Arrays for plot
        self.plotDataPSNR = []
        self.plotDataSSIM = []
        self.plotDataMSSIM = []
        self.plotDataVMAF = []
        self.plotDataSPSNR = []
        self.plotDataWSPSNR = []

        self.isPSNRevaluated = False
        self.isSSIMevaluated = False
        self.isMSSIMevaluated = False
        self.isVMAFevaluated = False
        self.isSPSNRevaluated = False
        self.isWSPSNRevaluated = False

        #Change window icon to QualiApp360's logo
        logo = PhotoImage(file='img/LOGO48_full.png')
        self.root.iconphoto(False, logo)

        #Disable resize of window
        self.root.resizable(width=False, height=False)

        '''LEFT SIDE: DATA GENERATOR'''
        self.nameLabel = Label(self.root,text="QualiApp360", width=20, font=myFont, bg="white",fg="#2189d1")
        self.nameLabel.config(font=("Nyata FTR", 35))
        self.nameLabel.place(x=0,y=157)


        self.currentDir = StringVar()
        
        #Img title
        appImg= Canvas(self.root, width = 300, height= 170, bg="white", bd=0, highlightbackground='white', highlightthickness=0)
        appImg.pack()
        imgTitle = PhotoImage(file = "img/title.png")
        appImg.create_image(0, 0, anchor=NW, image=imgTitle)
        appImg.place(x=100,y=2)
        
        #Img line separator
        lineImg= Canvas(self.root, width = 50, height= 585, bg="white", bd=0, highlightbackground='white', highlightthickness=0)
        lineImg.pack()
        imgLine = PhotoImage(file = "img/line.png")
        lineImg.create_image(0, 0, anchor=NW, image=imgLine)
        lineImg.place(x=480,y=5)
        
        #Text area Width
        self.input_widthLabel = Label(self.root,text="Width", bg="white", font=myFontLittle)
        self.input_widthLabel.place(x=100,y=242)

        #Label Width
        self.input_width = Text(height = 1, width = 7, bg="#E0F3FF")
        self.input_width.place(x=170,y=245)

        #Text area Height
        self.input_heightLabel = Label(self.root,text="Height", bg="white", font=myFontLittle)
        self.input_heightLabel.place(x=100,y=288)
        
        #Label Height
        self.input_height = Text(height = 1, width = 7, bg="#E0F3FF")
        self.input_height.place(x=170,y=290)

        #Text area Height
        self.input_framesLabel = Label(self.root,text="Frames", bg="white", font=myFontLittle)
        self.input_framesLabel.place(x=100,y=330)

        #Label Frames
        self.input_frames = Text(height = 1, width = 7, bg="#E0F3FF")
        self.input_frames.place(x=170,y=335)

        #Button Open File 1      
        self.buttonOpenFile1 = Button(self.root, text="Reference video", command=self.open_file1, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonOpenFile1.place(x=250,y=235)       
       
        #Label directory video 1
        checkImg = PhotoImage(file = "img/check.png")
        self.checkLabel1 = Label(self.root,image = checkImg, bg="white", font=myFont)
        self.checkLabel1.place(x=415,y=1000)
                
        #Button Open File 2       
        self.buttonOpenFile2 = Button(self.root, text="Impaired video", command=self.open_file2, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonOpenFile2.place(x=250,y=280) 
        self.buttonOpenFile2['state'] = 'disabled'

        #Label directory video 2
        self.checkLabel2 = Label(self.root,image = checkImg, bg="white", font=myFont)
        self.checkLabel2.place(x=415,y=1000)

                        
        #Button Open Spherical txt      
        self.buttonOpenFile3 = Button(self.root, text="Spherical txt", command=self.open_file3, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonOpenFile3.place(x=250,y=325)
        self.buttonOpenFile3['state'] = 'disabled'

        
        #Checkbox METRICS
        self.chk_psnr_state = BooleanVar()
        self.chk_ssim_state = BooleanVar()
        self.chk_mssim_state = BooleanVar()
        self.chk_vmaf_state = BooleanVar()
        self.chk_s_psnr_state = BooleanVar()
        self.chk_ws_psnr_state = BooleanVar()
        
        self.chk_psnr_state.set(False)
        self.chk_ssim_state.set(False)
        self.chk_mssim_state.set(False)
        self.chk_vmaf_state.set(False)
        self.chk_s_psnr_state.set(False)
        self.chk_ws_psnr_state.set(False)

        self.chk_psnr = Checkbutton(self.root, text='PSNR', var=self.chk_psnr_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_ssim = Checkbutton(self.root, text='SSIM', var=self.chk_ssim_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_mssim = Checkbutton(self.root, text='MSSIM', var=self.chk_mssim_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_vmaf = Checkbutton(self.root, text='VMAF 360º', var=self.chk_vmaf_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_s_psnr = Checkbutton(self.root, text='S-PSNR', var=self.chk_s_psnr_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_ws_psnr = Checkbutton(self.root, text='WS-PSNR', var=self.chk_ws_psnr_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        
        self.chk_psnr.place(x=110,y=370)
        self.chk_ssim.place(x=110,y=400)
        self.chk_mssim.place(x=110,y=430)
        self.chk_vmaf.place(x=280,y=370)
        self.chk_s_psnr.place(x=280,y=400)
        self.chk_ws_psnr.place(x=280,y=430)

        #All metrics disabled until both videos are loaded
        self.disableAllMetrics()
        
        #Button Start      
        self.buttonStart = Button(self.root, text="Start evaluation", command=self.start_process, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonStart.place(x=100,y=475)
        self.buttonStart['state'] = 'disabled'

        #Progress bar
        self.style.configure("blue.Horizontal.TProgressbar", background='#2189d1')
        self.progressBar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', style='blue.Horizontal.TProgressbar')
        self.progressBar.place(x=100, y=530, width=305)      

        #Label completed
        self.completedLabel = Label(self.root,text = "Evaluation completed!", bg="white", fg="#2189d1") 
        self.completedLabel.config(font=("Nyata FTR", 12))
        self.completedLabel.place(x=175,y=1000)
        

        '''RIGHT SIDE: DATA RECOVERY'''
        
        #Label choose metrics
        self.metricsLabel = Label(self.root,text=" Choose metrics to show on the graphic", width=35, bg="white", fg="#2189d1") 
        self.metricsLabel.config(font=("Nyata FTR", 17))
        self.metricsLabel.place(x=535,y=75)

        #Checkbox mu, sigma, max, min
        self.chk_mu_state = BooleanVar()
        self.chk_sigma_state = BooleanVar()
        self.chk_min_state = BooleanVar()
        self.chk_max_state = BooleanVar()

        self.chk_mu_state.set(True)
        self.chk_sigma_state.set(True)
        self.chk_min_state.set(True)
        self.chk_max_state.set(True)

        self.chk_mu = Checkbutton(self.root, text='\u03BC', var=self.chk_mu_state, bg="white", font=myFont)
        self.chk_sigma = Checkbutton(self.root, text='\u03C3', var=self.chk_sigma_state, bg="white", font=myFont)
        self.chk_min = Checkbutton(self.root, text='min', var=self.chk_min_state, bg="white", font=myFont)
        self.chk_max = Checkbutton(self.root, text='max', var=self.chk_max_state, bg="white", font=myFont)
                
        self.chk_mu.place(x=650,y=110)
        self.chk_sigma.place(x=700,y=110)
        self.chk_min.place(x=750,y=110)
        self.chk_max.place(x=815,y=110)

        # Disable all the elements from the graph side
        self.chk_mu['state'] = 'disabled'
        self.chk_sigma['state'] = 'disabled'
        self.chk_min['state'] = 'disabled'
        self.chk_max['state'] = 'disabled'
        '''
        #ComboBox Metrics
        self.comboMetrics = ttk.Combobox(self.root, state = "readonly")
        self.comboMetrics.place(x=700, y=150)
        self.comboMetrics["values"] = ["PSNR","SSIM","MSSIM","S-PSNR","WS-PSNR"]
        '''
        #Show Plot Buttons
        #Label choose metrics
        self.metricsLabel = Label(self.root,text="Plot output", width=35, bg="white", fg="#2189d1") 
        self.metricsLabel.config(font=("Nyata FTR", 17))
        self.metricsLabel.place(x=535,y=165)
        
        self.buttonShowPlotPSNR = Button(self.root, text="Show PSNR plot", command=self.showPSNR, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonShowPlotPSNR.place(x=590,y=210)
        self.buttonShowPlotPSNR['state'] = 'disabled'

        self.buttonShowPlotSSIM = Button(self.root, text="Show SSIM plot", command=self.showSSIM, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonShowPlotSSIM.place(x=590,y=270)
        self.buttonShowPlotSSIM['state'] = 'disabled'

        self.buttonShowPlotMSSIM = Button(self.root, text="Show MSSIM plot", command=self.showMSSIM, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonShowPlotMSSIM.place(x=590,y=330)
        self.buttonShowPlotMSSIM['state'] = 'disabled'

        self.buttonShowPlotVMAF = Button(self.root, text="Show VMAF plot", command=self.showVMAF, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonShowPlotVMAF.place(x=760,y=210)
        self.buttonShowPlotVMAF['state'] = 'disabled'

        self.buttonShowPlotSPSNR = Button(self.root, text="Show S-PSNR plot", command=self.showSPSNR, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonShowPlotSPSNR.place(x=760,y=270)
        self.buttonShowPlotSPSNR['state'] = 'disabled'
        
        self.buttonShowPlotWSPSNR = Button(self.root, text="Show WS-PSNR plot", command=self.showWSPSNR, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonShowPlotWSPSNR.place(x=760,y=330)
        self.buttonShowPlotWSPSNR['state'] = 'disabled'


        #Label save metrics
        self.metricsLabel = Label(self.root,text="Save the results", width=35, bg="white", fg="#2189d1") 
        self.metricsLabel.config(font=("Nyata FTR", 17))
        self.metricsLabel.place(x=535,y=400)

        #Button CSV      
        self.buttonSaveCSV = Button(self.root, text="Download CSV", command=self.save_csv, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonSaveCSV.place(x=600,y=445)
        self.buttonSaveCSV['state'] = 'disabled'

        #Button userManual
        helpImg = PhotoImage(file = "img/help_45.png")
        self.buttonManual = Button(self.root, image = helpImg, command=self.open_manual, bd=0, bg="white", cursor="hand2")
        self.buttonManual.place(x=950,y=550)
        
        self.root.mainloop()

    #Function open manual
    def open_manual(self):   
        subprocess.Popen("UserManual.pdf", shell=True)

    #Function start process
    def start_process(self):

        self.progressBar.start        
        self.frame_actual = 0
        
        self.buttonOpenFile2['state'] = 'disabled'        
        self.buttonStart['state'] = 'disabled'

        if(self.isPSNRchecked == True):
            self.isPSNRevaluated = True
        if(self.isSSIMchecked == True):
            self.isSSIMevaluated = True
        if(self.isMSSIMchecked == True):
            self.isMSSIMevaluated = True
        if(self.isVMAFchecked == True):
            self.isVMAFevaluated = True
        if(self.isSPSNRchecked == True):
            self.isSPSNRevaluated = True
        if(self.isWSPSNRchecked == True):
            self.isWSPSNRevaluated = True

        self.disableAllMetrics()


        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--delay", type=int, default=15, help=" Time delay")

        args = parser.parse_args()
        delay = args.delay
        framenum = -1 # Frame counter
        captRefrnc = self.first_video
        captUndTst = self.second_video
        totalWS = 0
        durationWS = 0

        if(self.dir1.endswith('.mp4') | self.dir1.endswith('.avi')):                    
            self.global_width = self.size_from_video[0]
            self.global_height = self.size_from_video[1]

        
        self.num_frames = self.num_frames_from_video 
        
        #START PROCESS
        print("Size: Width={} Height={} | Number of Frames#: {}".format(self.global_width, self.global_height, self.num_frames))

        #config progressBar
        self.progressBar.config(maximum= self.num_frames)
        
        #VMAF
        if(self.isVMAFchecked() == True):
            
            self.frame_actual = 0
            getVMAFdata(self)

        if(self.isWSPSNRchecked() == True):
            self.frame_actual = 0
            j = 0          
            
            weightMap = np.ones((self.global_width, self.global_height), dtype=np.float64)

            while(j < self.global_height):
                weight = cos((j - (self.global_height/2 - 0.5)) * math.pi / self.global_height)
                weightMap[j] = weightMap[j] * weight
                j+=1

            totalWS, durationWS = getWSPSNRdata(self, captRefrnc, captUndTst, self.num_frames, weightMap)

            #WS-PSNR data print
            global_ws_psnr = totalWS / self.num_frames
            avg_time_ws_psnr = durationWS / self.num_frames
            
            print("Global WS-PSNR: {}".format(round(global_ws_psnr, 4)), end=" ")
            print()
            print("Average Time: {}".format(round(avg_time_ws_psnr, 4)), end=" ")

        if(self.isSPSNRchecked() == True):
            self.frame_actual = 0
            j = 0      

            sum_SPSNR = getSPSNRdata(self, self.sph_points_x, self.sph_points_y, self.n_sphere, captRefrnc, captUndTst)

            #S-PSNR data print            
            print("S-PSNR: {}".format(round(sum_SPSNR, 4)), end=" ")
            print()
        
        #2D Metrics: PSNR, SSIM, MSSIM
        #if(self.isPSNRchecked() == True | self.isMSSIMchecked() == True | self.isSSIMchecked() == True):
        self.frame_actual = 0
        while (self.frame_actual < self.num_frames or (msvcrt.kbhit() and msvcrt.getch()[0] == 27)):           
            
            self.frame_actual += 1
            self.progressBar['value'] = self.frame_actual 
            self.root.update_idletasks()
            
            framenum += 1

            _, frameReference = captRefrnc.read()
            _, frameUnderTest = captUndTst.read()

            if frameReference is None or frameUnderTest is None:
                break
        
            if(self.isPSNRchecked() == True):
                psnrv = getPSNRdata(frameReference, frameUnderTest)
                print("Frame: {}# PSNR: {}dB".format(framenum, round(psnrv, 3)), end=" ")
                self.plotDataPSNR.append(psnrv)
                print()

            if(self.isMSSIMchecked() == True):
                mssimv = getMSSIMdata(frameReference, frameUnderTest)
                print("MSSIM: R {}% G {}% B {}%".format(round(mssimv[2] * 100, 2), round(mssimv[1] * 100, 2), round(mssimv[0] * 100, 2), end=" "))
                print()
                self.plotDataMSSIM.append(mssimv)

            if(self.isSSIMchecked() == True):
                ssim = getSSIMdata(frameReference, frameUnderTest)
                print("SSIM: {}".format(ssim, end=" "))
                print()
                self.plotDataSSIM.append(ssim)
                


            if cv.waitKey(1) & 0xFF == 27:
                break
        

        self.completedLabel.place(x=175,y=565)  

        self.enable_chk_graph()

        if(self.isPSNRchecked()):
            self.createPlotPSNR()
        if(self.isMSSIMchecked()):
            self.createPlotMSSIM()
        if(self.isSSIMchecked()):
            self.createPlotSSIM()
        if(self.isVMAFchecked()):
            self.createPlotVMAF()
        if(self.isSPSNRchecked()):
            self.createPlotSPSNR()
        if(self.isWSPSNRchecked()):
            self.createPlotWSPSNR()
 
    #Function create plots
    def createPlotPSNR(self):
       
        x = []
        y = []
        
        n_frames = int(self.num_frames_from_video)
        x = list(range(n_frames))
        y = self.plotDataPSNR

        plt.plot(x, y, linewidth=2.0)
        plt.axis([0,n_frames, 0, 100])
        plt.xlabel('Number of frames')
        plt.ylabel('PSNR')
        plt.title('PSNR results')

      #Function create plots
    def createPlotSSIM(self):
       
        x = []
        y = []
        
        n_frames = int(self.num_frames_from_video)
        x = list(range(n_frames))
        y = self.plotDataSSIM
        
        plt.plot(x, y, linewidth=2.0)
        plt.axis([0,n_frames, 0, 2])
        plt.xlabel('Number of frames')
        plt.ylabel('SSIM')
        plt.title('SSIM results')


      #Function create plots
    def createPlotMSSIM(self):
       
        x = []
        y = []
        
        n_frames = int(self.num_frames_from_video)
        x = list(range(n_frames))
        y = self.plotDataMSSIM
        
        plt.plot(x, y, linewidth=2.0)
        plt.axis([0,n_frames, 0, 2])
        plt.xlabel('Number of frames')
        plt.ylabel('MSSIM')
        plt.title('MSSIM results')

        red = mpatches.Patch(color='orange', label='R')
        green = mpatches.Patch(color='green', label='G')
        blue = mpatches.Patch(color='blue', label='B')

        plt.legend(handles=[red, green, blue])


  #Function create plots
    def createPlotVMAF(self):
       
        x = []
        y = []
        
        n_frames = int(self.num_frames_from_video)
        x = list(range(n_frames))
        y = self.plotDataVMAF
        
        plt.plot(x, y, linewidth=2.0)
        plt.axis([0,n_frames, 0, 100])
        plt.xlabel('Number of frames')
        plt.ylabel('VMAF')
        plt.title('VMAF results')



  #Function create plots
    def createPlotSPSNR(self):
       
        x = []
        y = []
        
        n_frames = int(self.num_frames_from_video)
        x = list(range(n_frames))
        y = self.plotDataSPSNR
        
        plt.plot(x, y, linewidth=2.0)
        plt.axis([0,n_frames, 0, 100])
        plt.xlabel('Number of frames')
        plt.ylabel('SPSNR')
        plt.title('SPSNR results')

  #Function create plots
    def createPlotWSPSNR(self):
       
        x = []
        y = []
        
        n_frames = int(self.num_frames_from_video)
        x = list(range(n_frames))
        y = self.plotDataSPSNR
        
        plt.plot(x, y, linewidth=2.0)
        plt.axis([0,n_frames, 0, 100])
        plt.xlabel('Number of frames')
        plt.ylabel('WSPSNR')
        plt.title('WSPSNR results')



    #Function open file 1
    def open_file1(self):

        self.disableAllMetrics() 

        if(messagebox.askokcancel("Warning","To evaluate spherical videos you need to add the required data before choosing a file. Do you want to continue?")):
            self.dir1 = filedialog.askopenfilename(initialdir="/",title="Select Video",
                filetypes=(("all files", "*.*"),("avi files","*.avi"),("mp4 files","*.mp4"),("yuv files","*.yuv")))

            if(self.dir1.endswith('.yuv') | self.dir1.endswith('.avi') | self.dir1.endswith('.mp4')):
                self.size_from_video = 0

                if(self.dir1.endswith('.yuv')):
                    try:
                        self.global_width = int(self.input_width.get("1.0", "end"))
                        self.global_height = int(self.input_height.get("1.0", "end"))
                        self.num_frames_from_video  = int(self.input_frames.get("1.0", "end"))
                        self.num_frames = self.num_frames_from_video 
                        self.first_video = VideoCaptureYUV(self, self.dir1, self.global_width, self.global_height)
                        self.check1 = True
                        self.buttonOpenFile2['state'] = 'normal'
                        self.buttonOpenFile3['state'] = 'normal'
                    except:
                        messagebox.showerror(message="Error loading video file. Please input the required data before loading a video", title="Error")
                else:
                    try:
                        self.first_video = cv.VideoCapture(self.dir1)
                        self.global_width = self.first_video.get(CAP_PROP_FRAME_WIDTH)
                        self.global_height = self.first_video.get(CAP_PROP_FRAME_HEIGHT)
                        self.size_from_video = (self.global_width, self.global_height)
                        self.num_frames_from_video = self.first_video.get(CAP_PROP_FRAME_COUNT)
                        self.check1 = True
                        self.buttonOpenFile2['state'] = 'normal'
                    except:
                        messagebox.showerror(message="Error loading video file.", title="Error")

            if(self.check1 == True & self.check2 == True):
                self.check1 = False
                self.check2 = False
                self.buttonStart['state'] = 'normal'
                
    #Function open file 2
    def open_file2(self):
        
        self.disableAllMetrics()   

        self.dir2 = filedialog.askopenfilename(initialdir="/",title="Select Video",
            filetypes=(("all files", "*.*"),("avi files","*.avi"),("mp4 files","*.mp4"),("yuv files","*.yuv")))
        
        if(self.dir2.endswith('.yuv') | self.dir2.endswith('.avi') | self.dir2.endswith('.mp4')):

            if(self.dir2.endswith('.yuv')):
                try:
                    self.second_video = VideoCaptureYUV(self, self.dir2, self.global_width, self.global_height)
                    self.check2 = True
                    self.enableAllMetrics()        
                except:
                    messagebox.showerror(message="Error loading video file.", title="Error")
            else:
                try:
                    self.second_video = cv.VideoCapture(self.dir2)
                    self.check2 = True
                    self.enableAllMetrics()                
                except:
                    messagebox.showerror(message="Error loading video file.", title="Error")

                #Checks if both videos are loaded
                if(self.check1 == True & self.check2 == True):
                    self.check1 = False
                    self.check2 = False
                    self.enableAllMetrics()
        else:
            messagebox.showerror(message="An error has occurred while choosing the video file.", title="Error")

     #Function open txt file
    def open_file3(self): 
        
        self.dir3 = filedialog.askopenfilename(initialdir="/",title="Select File",
            filetypes=[("txt files", "*.txt")])
        
        if(self.dir3.endswith('.txt')):
            self.n_sphere = 0

            with open(self.dir3, 'r') as data:
                self.sph_points_x = []
                self.sph_points_y = []
                for line in data:
                    p = line.split()
                    self.sph_points_x.append(p[0])
                    self.sph_points_y.append(p[1])
                    self.n_sphere += 1
            
            #self.sph_points = np.c_[sph_x, sph_y]
            #self.sph_points = [sph_x, sph_y]
            self.check3 = True
            self.enableAllMetrics()
        else:
            messagebox.showerror(message="Choose a txt file", title="Error")
        


    def enableAllMetrics(self):
        
        if (self.dir1.endswith('.mp4') | self.dir1.endswith('.avi')):
            self.chk_psnr['state'] = 'normal'
            self.chk_ssim['state'] = 'normal'
            self.chk_mssim['state'] = 'normal'

        if(self.dir1.endswith('.yuv')):
            self.chk_vmaf['state'] = 'normal'
            self.chk_ws_psnr['state'] = 'normal'
            
            if(self.check3 == True):
                self.chk_s_psnr['state'] = 'normal'


    def disableAllMetrics(self):
        self.chk_psnr['state'] = 'disabled'
        self.chk_ssim['state'] = 'disabled'
        self.chk_mssim['state'] = 'disabled'
        self.chk_vmaf['state'] = 'disabled'
        self.chk_s_psnr['state'] = 'disabled'
        self.chk_ws_psnr['state'] = 'disabled'


    #Function activate checkboxes and buttons from the graph side
    def enable_chk_graph(self):
        self.chk_mu['state'] = 'normal'
        self.chk_sigma['state'] = 'normal'
        self.chk_min['state'] = 'normal'
        self.chk_max['state'] = 'normal'
        
        if self.isPSNRchecked():                
            self.buttonShowPlotPSNR['state'] = 'normal'
        if self.isSSIMchecked():    
            self.buttonShowPlotSSIM['state'] = 'normal'
        if self.isMSSIMchecked():    
            self.buttonShowPlotMSSIM['state'] = 'normal'
        if self.isVMAFchecked():    
            self.buttonShowPlotVMAF['state'] = 'normal'
        if self.isSPSNRchecked():    
            self.buttonShowPlotSPSNR['state'] = 'normal'
        if self.isWSPSNRchecked():    
            self.buttonShowPlotWSPSNR['state'] = 'normal'

        self.buttonSaveCSV['state'] = 'normal'

    def showPSNR(self):
        plt.show()

    def showSSIM(self):
        plt.show()
    
    def showMSSIM(self):
        plt.show()

    def showVMAF(self):
        plt.show()
    
    def showSPSNR(self):
        plt.show()
    
    def showWSPSNR(self):
        plt.show()
    
    #Function save data on a CSV file
    def save_csv(self):
        messagebox.showinfo(message="Selected metrics will be downloaded. Existing files will be overwritten.", title="")

        if self.isPSNRchecked(): 
            i = 1
            n_frames = int(self.num_frames_from_video)
            
            with open('psnr_metrics.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                while(i < n_frames):
                    rowlist = [self.plotDataPSNR[i]]
                    writer.writerow(rowlist)
                    i += 1

        if self.isSSIMchecked(): 
            i = 1
            n_frames = int(self.num_frames_from_video)
            
            with open('ssim_metrics.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                while(i < n_frames):
                    rowlist = [self.plotDataSSIM[i]]
                    writer.writerow([rowlist])
                    i += 1

        if self.isMSSIMchecked(): 
            i = 1
            n_frames = int(self.num_frames_from_video)
            
            with open('mssim_metrics.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                while(i < n_frames):
                    rowlist = [self.plotDataMSSIM[i]]
                    writer.writerow(rowlist)
                    i += 1

        if self.isSPSNRchecked(): 
            i = 1
            n_frames = int(self.num_frames_from_video)
            
            with open('spsnr_metrics.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                while(i < n_frames):
                    rowlist = [self.plotDataSPSNR[i]]
                    writer.writerow(rowlist)
                    i += 1
                    
    def isPSNRchecked(self):
        if(self.chk_psnr_state.get()):
            return True
        else: 
            return False

    def isMSSIMchecked(self):
        if(self.chk_mssim_state.get()):
            return True
        else: 
            return False

    def isSSIMchecked(self):
        if(self.chk_ssim_state.get()):
            return True
        else: 
            return False

    def isVMAFchecked(self):
        if(self.chk_vmaf_state.get()):
            return True
        else: 
            return False

    def isSPSNRchecked(self):
        if(self.chk_s_psnr_state.get()):
            return True
        else: 
            return False

    def isWSPSNRchecked(self):
        if(self.chk_ws_psnr_state.get()):
            return True
        else: 
            return False


    def isAtLeastOneChecked(self):
        if(self.isWSPSNRchecked() | self.isSPSNRchecked()| self.isVMAFchecked() | self.isSSIMchecked() |  self.isMSSIMchecked() |  self.isPSNRchecked()): 
            self.buttonStart['state'] = 'normal'
        else:
            self.buttonStart['state'] = 'disabled'


def getVMAFdata(self):
    parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser(description='VMAF ODV')
    
    parser.add_argument('--w', default=self.global_width, help="resolution width of a given videos")
    parser.add_argument('--h', default=self.global_height, help="resolution height of a given videos")
    parser.add_argument('--f', default=self.num_frames, help="number of frame")
    parser.add_argument('--r', default=self.dir1,  help="reference video")
    parser.add_argument('--c', nargs='?', type=int, default=15, action="store", help="cell number")

    user_input = parser.parse_args()
    
    user_input.r = os.path.basename(self.dir1)[:-4]
   # download the zip file and extract it
    try:
        if(os.path.isfile(file_name)!=True):
            wget.download('http://v-sense.scss.tcd.ie/Datasets/' + file_name, file_name)
            extract_process(file_name)
        if(os.path.isfile(thirdParty_zip)!=True):
            wget.download('http://v-sense.scss.tcd.ie/Datasets/' + thirdParty_zip, thirdParty_zip)
            extract_process(thirdParty_zip)
    except:
        print("No file to be downloaded!")

    # generate patches for each video file
    try:
        types = (video_folder + '*.mp4', video_folder + '*.yuv')
        files =[]
        [files.extend(glob.glob(_type)) for _type in types]
        for video in files:
            # convert mp4 to yuv
            #mp42yuv(user_input, video)
            # create a folder for each video
            create_dir(video_folder + 'results/' + os.path.basename(video)[:-4] + '/')
            # generate xml file for each video settings
            #xml_created(video, user_input)
            # generate patches
            generate_patches(video)
    except:
        print()
   
    # compute vmaf scores
    try:
        for video in glob.glob(video_folder + '*.yuv'):
            if os.path.basename(video)[:-4] != user_input.r:
                # compute vmaf for erp
                compute_vmafScores(user_input, video, user_input.r)
                # report the results
                report_vmafScores(video)
                for patch in glob.glob( video_folder + 'results/' + os.path.basename(video)[:-4] + '/*.yuv'):
                    # compute vmaf per patch
                    compute_patchScores(video, patch, user_input.r)
    except:
        print()

    # report and clean the project
    try:  
        for video in glob.glob(video_folder + '*.yuv'):
            if os.path.basename(video)[:-4] != user_input.r:
                agg_result = {}
                for patch in glob.glob( video_folder + 'results/' + os.path.basename(video)[:-4] + '/*.xml'):
                    #remove yuv file
                    remove_file(patch[:-4])
                    agg_result[os.path.basename(patch)[:-4]] = report_results(video, patch)
                rows = [agg_result[x] for x in agg_result.keys()]
                #for python2 delete newline and add wb
                # with open(os.path.basename(video)[:-4] + '.csv', 'wb') as f:
                with open( 'vi_vmaf_' + os.path.basename(video)[:-4] + '.csv', 'w', newline="") as f:
                    w = csv.writer(f)

                    w.writerow([vmaf_model])
                    w.writerow([os.path.basename(video)])
                    w.writerow([user_input.r + '.mp4'])
                    _vor_vmaf = []
                    # import pdb; pdb.set_trace()
                    for f in range(len(rows[0])):
                        row = [rows[l][f] for l in range(len(agg_result.keys()))]                        
                        vor_vmaf = [float(rows[l][f]) for l in range(len(agg_result.keys()))]
                        vor_vmaf = sum(vor_vmaf)/len(agg_result.keys())
                        _vor_vmaf.append(vor_vmaf)

                    avg_vmaf = sum(_vor_vmaf)/len(_vor_vmaf)
                    print("::.. 360 VI-VMAF score for {} = {}".format(os.path.basename(video)[:-4], round(avg_vmaf,4)))
                    w.writerow(['VI-VMAF: '+ str(round(avg_vmaf,4))])
                    [w.writerow([round(score_per_frame,4)]) for score_per_frame in _vor_vmaf]
                   
            else:
                for patch in glob.glob( video_folder + 'results/' + os.path.basename(video)[:-4] + '/*.yuv'):
                    remove_file(patch[:-4])

    except:
        print()


def width_height_from_str(s):
    m = re.search(".*[_-](\d+)x(\d+).*", s)
    if not m:
        print ("Could not find resolution in file name: %s" % (s))
        exit(1)

    w = int(m.group(1))
    h = int(m.group(2))
    return w,h

def remove_file(_file):
    try:
        os.remove(_file + ".yuv")
    except:
        pass

def create_dir(folder):
    try:
        os.makedirs(folder)
    except:
        pass

def extract_process(name):
    # opening the zip file in READ mode
    with ZipFile(name, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()
 
        # extracting all the files
        print('Extracting all the files now...')
        zip.extractall(project_name + '/')
        print('Done!')


def report_results(video, patch):
    '''
    Step 3: Report the results
    '''
    result_patch    = video_folder + 'results/' + os.path.basename(video)[:-4] + '/' + os.path.basename(patch)[:-4] + '.xml'
    doc = ET.parse(result_patch)
    root = doc.getroot()
    res = []
    for elem in root.iter('frame'):
        res.append(elem.attrib['vmaf'])
    return res

def compute_patchScores(video, patch, ref):
    '''
    Step 2: Computing the Voronoi patch scores
    '''

    dis_patch       = video_folder + 'results/' + os.path.basename(video)[:-4] + '/' + os.path.basename(patch)
    ref_patch       = video_folder + 'results/' + ref + '/' + os.path.basename(patch)
    height_patch    = width_height_from_str(ref_patch)[1]
    width_patch     = width_height_from_str(ref_patch)[0]
    result_patch    = video_folder + 'results/' + os.path.basename(video)[:-4] + '/' + os.path.basename(patch)[:-4] + '.xml'


    cmd  = project_name + 'vmafossexec yuv420p ' + str(width_patch) + ' ' + str(height_patch) + ' ' + ref_patch + ' ' \
    + dis_patch + ' ' + project_name + 'model/' + vmaf_model + ' --log ' + result_patch + ' --log_fmt csv --psnr --ssim --ms-ssim --thread 0 --subsample 1 --ci'
    cmd = cmd.replace("/","\\")
    if(os.path.isfile(result_patch)!=True):
        os.system(cmd)

def compute_vmafScores(user_input, video, ref):
   
    result      = video_folder + 'results/' + os.path.basename(video)[:-4] + '/' + os.path.basename(video)[:-4] + '.xml'
   
    cmd  = project_name + 'vmafossexec yuv420p ' + str(user_input.w) + ' ' + str(user_input.h) + ' ' + video_folder + ref + '.yuv' + ' ' \
    + video_folder + os.path.basename(video)[:-3] + 'yuv' + ' ' + project_name + 'model/' + vmaf_model + ' --log ' + result + ' --log_fmt csv --psnr --ssim --ms-ssim --thread 0 --subsample 1 --ci'
    cmd = cmd.replace("/","\\")

    if(os.path.isfile(result)!=True):
        os.system(cmd)

def generate_patches(video):
    '''
    Step 1: Generate Voronoi patches
    '''
    cmd = project_name + vpatch + ' ' + video_folder + 'results/' + os.path.basename(video)[:-4] + '.xml'
    cmd = cmd.replace("/","\\")
    #if(os.path.isfile(video_folder + 'results/' + os.path.basename(video)[:-4] + '.xml')!=True):
    os.system(cmd)

def report_vmafScores(self, video):

    _vmaf = report_results(video, video)
    with open( 'vmaf_' + os.path.basename(video)[:-4] + '.csv', 'w', newline="") as f:
        w = csv.writer(f)
        w.writerow([vmaf_model])
        w.writerow([os.path.basename(video)])
        w.writerow([self.first_video + '.mp4'])
        __vmaf = [float(_vmaf[l]) for l in range(len(_vmaf))]
       
        avg_vmaf = sum(__vmaf)/len(__vmaf)

        
        self.frame_actual += 1
        self.progressBar['value'] = self.frame_actual 
        self.root.update_idletasks()
        
        self.plotDataVMAF.append(avg_vmaf)       
        print("::.. VMAF score for {} = {}".format(os.path.basename(video)[:-4], round(avg_vmaf,4)))
        w.writerow(['VMAF: '+ str(round(avg_vmaf,4))])
        [w.writerow([score_per_frame]) for score_per_frame in _vmaf]


def read_raw(self):
    self.frame_len = int(self.frame_len)
    
    raw = readRAW(self.frame_len)
    yuv = np.frombuffer(raw, dtype=np.uint8)
    yuv = yuv.reshape(self.shape)
    
    return True, yuv

def readRAW(self):
    ret, yuv = read_raw()
    if not ret:
        return ret, yuv
        
    bgr = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_I420)
    return ret, bgr


if __name__=="__main__":
    QualiApp360()