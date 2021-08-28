#IMPORTS NEEDED
from tkinter import * 
from tkinter import ttk
from tkinter import filedialog
import tkinter.font as tkf
import subprocess
from cv2 import *
import cv2 as cv
from numpy import * 
import numpy as np
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
        self.progressBar['value'] = total_frames
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
    self.height = height
    self.width = width
    self.frame_len = self.width * self.height * 3 / 2
    # Open '*.yuv' as a binary file.
    self.f = open(filename, 'rb')
    self.shape = (int(self.height*1.5), self.width)


class QualiApp360:
    def __init__(self):

        self.root = Tk()
        self.root.title("QualiApp360")
        self.root.geometry("1000x620")
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
        lineImg= Canvas(self.root, width = 50, height= 635, bg="white", bd=0, highlightbackground='white', highlightthickness=0)
        lineImg.pack()
        imgLine = PhotoImage(file = "img/line.png")
        lineImg.create_image(0, 0, anchor=NW, image=imgLine)
        lineImg.place(x=480,y=5)

        #Button userManual
        helpImg = PhotoImage(file = "img/help_45.png")
        self.buttonManual = Button(self.root, image = helpImg, command=self.open_manual, bd=0, bg="white", cursor="hand2")
        self.buttonManual.place(x=950,y=570)
        
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

        #Label directory video 2
        self.checkLabel2 = Label(self.root,image = checkImg, bg="white", font=myFont)
        self.checkLabel2.place(x=415,y=1000)

                        
        #Button Open Spherical txt      
        self.buttonOpenFile3 = Button(self.root, text="Spherical txt", command=self.open_file3, width= 15, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonOpenFile3.place(x=250,y=325)

        
        #Checkbox METRICS
        self.chk_psnr_state = BooleanVar()
        self.chk_ssim_state = BooleanVar()
        self.chk_mssim_state = BooleanVar()
        self.chk_vmaf_state = BooleanVar()
        self.chk_cpp_psnr_state = BooleanVar()
        self.chk_s_psnr_state = BooleanVar()
        self.chk_ws_psnr_state = BooleanVar()
        self.chk_ov_psnr_state = BooleanVar()
        
        self.chk_psnr_state.set(False)
        self.chk_ssim_state.set(False)
        self.chk_mssim_state.set(False)
        self.chk_vmaf_state.set(False)
        self.chk_cpp_psnr_state.set(False)
        self.chk_s_psnr_state.set(False)
        self.chk_ws_psnr_state.set(False)
        self.chk_ov_psnr_state.set(False)

        self.chk_psnr = Checkbutton(self.root, text='PSNR', var=self.chk_psnr_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_ssim = Checkbutton(self.root, text='SSIM', var=self.chk_ssim_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_mssim = Checkbutton(self.root, text='MSSIM', var=self.chk_mssim_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_vmaf = Checkbutton(self.root, text='VMAF 360º', var=self.chk_vmaf_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_cpp_psnr = Checkbutton(self.root, text='CPP-PSNR', var=self.chk_cpp_psnr_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_s_psnr = Checkbutton(self.root, text='S-PSNR', var=self.chk_s_psnr_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_ws_psnr = Checkbutton(self.root, text='WS-PSNR', var=self.chk_ws_psnr_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        self.chk_ov_psnr = Checkbutton(self.root, text='OV-PSNR', var=self.chk_ov_psnr_state, bg="white", font=myFont, command=self.isAtLeastOneChecked)
        
        self.chk_psnr.place(x=110,y=370)
        self.chk_ssim.place(x=110,y=400)
        self.chk_mssim.place(x=110,y=430)
        self.chk_vmaf.place(x=110,y=460)
        self.chk_cpp_psnr.place(x=280,y=370)
        self.chk_s_psnr.place(x=280,y=400)
        self.chk_ws_psnr.place(x=280,y=430)
        self.chk_ov_psnr.place(x=280,y=460)

        #All metrics disabled until both videos are loaded
        self.disableAllMetrics()
        
        #Button Start      
        self.buttonStart = Button(self.root, text="Start evaluation", command=self.start_process, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonStart.place(x=100,y=510)
        self.buttonStart['state'] = 'disabled'

        #Progress bar
        self.style.configure("blue.Horizontal.TProgressbar", background='#2189d1')
        self.progressBar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', style='blue.Horizontal.TProgressbar')
        self.progressBar.place(x=100, y=565, width=305)      

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
        
        #Button CSV      
        self.buttonSaveCSV = Button(self.root, text="Download CSV", command=self.save_csv, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonSaveCSV.place(x=600,y=430)
        self.buttonSaveCSV['state'] = 'disabled'
        
        '''
        #Button JSON      
        self.buttonSaveJSON = Button(self.root, text="Download JSON", command=self.save_json, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonSaveJSON.place(x=600,y=480)
        self.buttonSaveJSON['state'] = 'disabled'''
        
        self.root.mainloop()

    #Function open manual
    def open_manual(self):   
        subprocess.Popen("UserManual.pdf", shell=True)

    #Function start process
    def start_process(self):

        self.progressBar.start        
        self.frame_actual = 0
        
        self.buttonStart['state'] = 'disabled'
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
            num_frames = self.num_frames_from_video

     
        #get the frames from the input textbox
        num_frames = int(self.input_frames.get("1.0", "end"))
        
        #START PROCESS
        print("Size: Width={} Height={} | Number of Frames#: {}".format(self.global_width, self.global_height, num_frames))

        #config progressBar
        self.progressBar.config(maximum= num_frames)
        
        #VMAF
        if(self.isVMAFchecked() == True):
            
            self.frame_actual = 0
            getVMAFdata(self, num_frames)

        if(self.isWSPSNRchecked() == True):
            self.frame_actual = 0
            j = 0          
            
            weightMap = np.ones((self.global_width, self.global_height), dtype=np.float64)

            while(j < self.global_height):
                weight = cos((j - (self.global_height/2 - 0.5)) * math.pi / self.global_height)
                weightMap[j] = weightMap[j] * weight
                j+=1

            totalWS, durationWS = getWSPSNRdata(self, captRefrnc, captUndTst, num_frames, weightMap)

            #WS-PSNR data print
            global_ws_psnr = totalWS / num_frames
            avg_time_ws_psnr = durationWS / num_frames
            
            print("Global WS-PSNR: {}".format(round(global_ws_psnr, 4)), end=" ")
            print()
            print("Average Time: {}".format(round(avg_time_ws_psnr, 4)), end=" ")
        
        else: 
            self.frame_actual = 0
            #PSNR, SSIM, MSSIM, WS-PSNR
            while (self.frame_actual < num_frames):           
                
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
                    print()

                if(self.isMSSIMchecked() == True):
                    mssimv = getMSSIMdata(frameReference, frameUnderTest)
                    print("MSSIM: R {}% G {}% B {}%".format(round(mssimv[2] * 100, 2), round(mssimv[1] * 100, 2), round(mssimv[0] * 100, 2), end=" "))
                    print()

                if(self.isSSIMchecked() == True):
                    ssim = getSSIMdata(frameReference, frameUnderTest)
                    print("SSIM: {}".format(ssim, end=" "))
                    print()


                if cv.waitKey(1) & 0xFF == 27:
                    break
            

        self.completedLabel.place(x=175,y=585)  

        self.enable_chk_graph()      
 

    #Function open file 1
    def open_file1(self):
        self.dir1 = filedialog.askopenfilename(initialdir="/",title="Select Video",
            filetypes=(("all files", "*.*"),("avi files","*.avi"),("mp4 files","*.mp4"),("yuv files","*.yuv")))
        
        self.global_width = int(self.input_width.get("1.0", "end"))
        self.global_height = int(self.input_height.get("1.0", "end"))

        if(self.dir1.endswith('.yuv')):
            self.first_video = VideoCaptureYUV(self, self.dir1, self.global_width, self.global_height)
        else:
            self.first_video = cv.VideoCapture(self.dir1)
            size_from_video = (int(self.first_video.get(CAP_PROP_FRAME_WIDTH)), int(self.first_video.get(CAP_PROP_FRAME_HEIGHT)))
            num_frames_from_video = self.first_video.get(CAP_PROP_FRAME_COUNT)

        
        self.check1 = True
        self.checkLabel1.place(x=415,y=230) 

        if(self.check1 == True & self.check2 == True):
            self.check1 = False
            self.check2 = False
            self.buttonStart['state'] = 'normal'
            self.enableAllMetrics()


    #Function open file 2
    def open_file2(self):
        self.dir2 = filedialog.askopenfilename(initialdir="/",title="Select Video",
            filetypes=(("all files", "*.*"),("avi files","*.avi"),("mp4 files","*.mp4"),("yuv files","*.yuv")))
        
        if(self.dir2.endswith('.yuv')):
            self.second_video = VideoCaptureYUV(self, self.dir2, self.global_width, self.global_height)
        else:
            self.second_video = cv.VideoCapture(self.dir2)

        
        self.check2 = True
        self.checkLabel2.place(x=415,y=275) 

        self.enableAllMetrics()

        #Checks if both videos are loaded
        if(self.check1 == True & self.check2 == True):
            self.check1 = False
            self.check2 = False
            self.enableAllMetrics()

     #Function open txt file
    def open_file3(self): 
        self.dir3 = filedialog.askopenfilename(initialdir="/",title="Select txt file",
            filetypes=(("txt files", "*.txt")))       
        
        
        self.enableAllMetrics()
        self.checkLabel3.place(x=415,y=320) 


    def enableAllMetrics(self):
        
        if (self.dir1.endswith('.mp4') | self.dir1.endswith('.avi')):
            self.chk_psnr['state'] = 'normal'
            self.chk_ssim['state'] = 'normal'
            self.chk_mssim['state'] = 'normal'

        if(self.dir1.endswith('.yuv')):
            self.chk_vmaf['state'] = 'normal'
            self.chk_cpp_psnr['state'] = 'normal'
            self.chk_ws_psnr['state'] = 'normal'
            
            if(self.check3 == True):
                self.chk_s_psnr['state'] = 'normal'
            self.chk_ov_psnr['state'] = 'normal'


    def disableAllMetrics(self):
        self.chk_psnr['state'] = 'disable'
        self.chk_ssim['state'] = 'disable'
        self.chk_mssim['state'] = 'disable'
        self.chk_vmaf['state'] = 'disable'
        self.chk_cpp_psnr['state'] = 'disable'
        self.chk_s_psnr['state'] = 'disable'
        self.chk_ws_psnr['state'] = 'disable'
        self.chk_ov_psnr['state'] = 'disable'


    #Function activate checkboxes and buttons from the graph side
    def enable_chk_graph(self):
        self.chk_mu['state'] = 'normal'
        self.chk_sigma['state'] = 'normal'
        self.chk_min['state'] = 'normal'
        self.chk_max['state'] = 'normal'

    #Function save data on a CSV file
    def save_csv(self):
        print("csv")

        '''
        for cellObj in sheet['A1':'C3']:
            for cell in cellObj:
                    print(cell.coordinate, cell.value)
            print('--- END ---')
        
        OUTPUT:
            A1 ID
            B1 AGE
            C1 SCORE
            --- END ---
            A2 1
            B2 22
            C2 5
            --- END ---
            A3 2
            B3 15
            C3 6
            --- END ---
        '''

    #Function save data on a JSON file
    def save_json(self):
        print("json")


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

    def isCPPPSNRchecked(self):
        if(self.chk_cpp_psnr_state.get()):
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

    def isOVPSNRchecked(self):
        if(self.chk_ov_psnr_state.get()):
            return True
        else: 
            return False

    def isAtLeastOneChecked(self):
        if(self.isOVPSNRchecked() | self.isWSPSNRchecked() | self.isSPSNRchecked() | self.isCPPPSNRchecked() 
        | self.isVMAFchecked() | self.isSSIMchecked() |  self.isMSSIMchecked() |  self.isPSNRchecked()): 
            self.buttonStart['state'] = 'normal'
        else:
            self.buttonStart['state'] = 'disabled'


def getVMAFdata(self, num_frames):
    parser = argparse.ArgumentParser()

    
    parser = argparse.ArgumentParser(description='VMAF ODV')
    '''
    parser.add_argument('--w', required=True, action="store", help="resolution width of a given videos")
    parser.add_argument('--h', required=True, action="store", help="resolution height of a given videos")
    parser.add_argument('--f', required=True, action="store", help="number of frame")
    parser.add_argument('--r', required=True, action="store", help="reference video")
    parser.add_argument('--c', nargs='?', type=int, default=15, action="store", help="cell number")
    '''
    
    parser.add_argument('--w', default=self.global_width, help="resolution width of a given videos")
    parser.add_argument('--h', default=self.global_height, help="resolution height of a given videos")
    parser.add_argument('--f', default=num_frames, help="number of frame")
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