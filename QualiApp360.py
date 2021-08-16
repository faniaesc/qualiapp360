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

class QualiApp360:
    def __init__(self):

        self.root = Tk()
        self.root.title("QualiApp360")
        self.root.geometry("1000x600")
        self.root.configure(bg="white")
        myFont = tkf.Font(family="Nyata FTR", size=14)
        self.root.firstVideoCheck = False
        self.style = ttk.Style()
        self.style.theme_use('default')
        
        #Checks for Start Evaluation
        self.check1 = False
        self.check2 = False


        #Change window icon to QualiApp360's logo
        logo = PhotoImage(file='img/LOGO48_full.png')
        self.root.iconphoto(False, logo)

        #Disable resize of window
        self.root.resizable(width=False, height=False)

        '''LEFT SIDE: DATA GENERATOR'''
        self.nameLabel = Label(self.root,text="QualiApp360", width=20, font=myFont, bg="white",fg="#2189d1")
        self.nameLabel.config(font=("Nyata FTR", 35))
        self.nameLabel.place(x=0,y=170)


        self.currentDir = StringVar()
        
        #Img title
        appImg= Canvas(self.root, width = 300, height= 170, bg="white", bd=0, highlightbackground='white', highlightthickness=0)
        appImg.pack()
        imgTitle = PhotoImage(file = "img/title.png")
        appImg.create_image(0, 0, anchor=NW, image=imgTitle)
        appImg.place(x=100,y=15)
        
        #Img line separator
        lineImg= Canvas(self.root, width = 50, height= 585, bg="white", bd=0, highlightbackground='white', highlightthickness=0)
        lineImg.pack()
        imgLine = PhotoImage(file = "img/line.png")
        lineImg.create_image(0, 0, anchor=NW, image=imgLine)
        lineImg.place(x=480,y=5)

        #Button userManual
        helpImg = PhotoImage(file = "img/help_45.png")
        self.buttonManual = Button(self.root, image = helpImg, command=self.open_manual, bd=0, bg="white", cursor="hand2")
        self.buttonManual.place(x=0,y=555)
       
        #Button Open File 1      
        self.buttonOpenFile1 = Button(self.root, text="Open first video", command=self.open_file1, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonOpenFile1.place(x=100,y=265) 
      
       
        #Label directory video 1
        checkImg = PhotoImage(file = "img/check.png")
        self.checkLabel1 = Label(self.root,image = checkImg, bg="white", font=myFont)
        self.checkLabel1.place(x=415,y=1000)
                
        #Button Open File 2       
        self.buttonOpenFile2 = Button(self.root, text="Open second video", command=self.open_file2, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonOpenFile2.place(x=100,y=330)

        #Label directory video 2
        self.checkLabel2 = Label(self.root,image = checkImg, bg="white", font=myFont)
        self.checkLabel2.place(x=415,y=1000)

        #Progress bar
        self.style.configure("blue.Horizontal.TProgressbar", background='#2189d1')
        self.progressBar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate', style='blue.Horizontal.TProgressbar')
        self.progressBar.place(x=100, y=490, width=305)        
        
        #Button Start      
        self.buttonStart = Button(self.root, text="Start evaluation", command=self.start_process, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonStart.place(x=100,y=430)
        self.buttonStart['state'] = 'disabled'

        #Label completed
        self.completedLabel = Label(self.root,text = "Evaluation completed!", bg="white", fg="#2189d1") 
        self.completedLabel.config(font=("Nyata FTR", 12))
        self.completedLabel.place(x=175,y=1000)
        

        '''RIGHT SIDE: DATA RECOVERY'''
        
        #Label choose metrics
        self.metricsLabel = Label(self.root,text=" Choose metrics to save", width=25, bg="white", fg="#2189d1") 
        self.metricsLabel.config(font=("Nyata FTR", 17))
        self.metricsLabel.place(x=605,y=25)

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
                
        self.chk_mu.place(x=650,y=60)
        self.chk_sigma.place(x=700,y=60)
        self.chk_min.place(x=750,y=60)
        self.chk_max.place(x=815,y=60)

        #Checkbox PSNR and SSIM
        self.chk_psnr_state = BooleanVar()
        self.chk_ssim_state = BooleanVar()

        self.chk_psnr_state.set(True)
        self.chk_ssim_state.set(True)

        self.chk_psnr = Checkbutton(self.root, text='PSNR', var=self.chk_psnr_state, bg="white", font=myFont)
        self.chk_ssim = Checkbutton(self.root, text='SSIM', var=self.chk_ssim_state, bg="white", font=myFont)
        
        self.chk_psnr.place(x=680,y=90)
        self.chk_ssim.place(x=770,y=90)

        # Disable all the elements from the data recovery side
        self.chk_mu['state'] = 'disabled'
        self.chk_sigma['state'] = 'disabled'
        self.chk_min['state'] = 'disabled'
        self.chk_max['state'] = 'disabled'
        self.chk_psnr['state'] = 'disabled'
        self.chk_ssim['state'] = 'disabled'
        
        #Button CSV      
        self.buttonSaveCSV = Button(self.root, text="Download CSV", command=self.save_csv, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonSaveCSV.place(x=600,y=430)
        self.buttonSaveCSV['state'] = 'disabled'

        #Button JSON      
        self.buttonSaveJSON = Button(self.root, text="Download JSON", command=self.save_json, width= 30, bd=0, bg="#51bfe4", cursor="hand2", font=myFont)
        self.buttonSaveJSON.place(x=600,y=480)
        self.buttonSaveJSON['state'] = 'disabled'

        self.root.mainloop()

    #Function open manual
    def open_manual(self):   
        subprocess.Popen("UserManual.pdf", shell=True)

    #Function start process
    def start_process(self):
        self.progressBar.start        
        self.frame_actual = 0
        
        self.buttonStart['state'] = 'disabled'

        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--delay", type=int, default=15, help=" Time delay")
        parser.add_argument("-v", "--psnrtriggervalue", type=int, default=30, help="PSNR Trigger Value")
        #parser.add_argument("-r", "--ref", type=str, default="Megamind.avi", help="Path to reference video")
        #parser.add_argument("-t", "--undertest", type=str, default="Megamind_bugy.avi", help="Path to the video to be tested")

        args = parser.parse_args()
        #sourceReference = args.ref
        #sourceCompareWith = args.undertest
        sourceReference = self.first_video
        sourceCompareWith = self.second_video
        delay = args.delay
        psnrTriggerValue = args.psnrtriggervalue
        framenum = -1 # Frame counter
        captRefrnc = self.first_video
        captUndTst = self.second_video

        if not captRefrnc.isOpened():
            print("Could not open the reference " + sourceReference)
            sys.exit(-1)

        if not captUndTst.isOpened():
            print("Could not open case test " + sourceCompareWith)
            sys.exit(-1)

        refS = (int(captRefrnc.get(CAP_PROP_FRAME_WIDTH)), int(captRefrnc.get(CAP_PROP_FRAME_HEIGHT)))
        uTSi = (int(captUndTst.get(CAP_PROP_FRAME_WIDTH)), int(captUndTst.get(CAP_PROP_FRAME_HEIGHT)))

        if refS != uTSi:
            print("Inputs have different size!!! Closing.")
            sys.exit(-1)
        
        print("Reference frame resolution: Width={} Height={} of nr#: {}".format(refS[0], refS[1],captRefrnc.get(CAP_PROP_FRAME_COUNT)))

        self.frame_cont = int(captRefrnc.get(CAP_PROP_FRAME_COUNT))
        self.progressBar.config(maximum= self.frame_cont)

        print("PSNR trigger value {}".format(psnrTriggerValue))

        while (self.frame_actual < self.frame_cont):           
            
            self.frame_actual += 1
            self.progressBar['value'] = self.frame_actual 
            self.root.update_idletasks()

            _, frameReference = captRefrnc.read()
            _, frameUnderTest = captUndTst.read()

            if frameReference is None or frameUnderTest is None:
                break
            
            framenum += 1
            psnrv = getPSNRdata(frameReference, frameUnderTest)

            print("Frame: {}# {}dB".format(framenum, round(psnrv, 3)), end=" ")

            if (psnrv < psnrTriggerValue and psnrv):
                mssimv = getMSSIMdata(frameReference, frameUnderTest)
                print("MSSIM: R {}% G {}% B {}%".format(round(mssimv[2] * 100, 2), round(mssimv[1] * 100, 2), round(mssimv[0] * 100, 2), end=" "))
            
            print()


            k = waitKey(delay)
            if k == 27:
                break

        self.completedLabel.place(x=175,y=515)  

        self.activate_data()      

    #Function activate checkboxes and buttons from the data recovery side
    def activate_data(self):
        # Enable all the elements from the data recovery side
        self.chk_mu['state'] = 'normal'
        self.chk_sigma['state'] = 'normal'
        self.chk_min['state'] = 'normal'
        self.chk_max['state'] = 'normal'
        self.chk_psnr['state'] = 'normal'
        self.chk_ssim['state'] = 'normal'


    #Function open file 1
    def open_file1(self):
        self.dir = filedialog.askopenfilename(initialdir="/",title="Select Video",
         filetypes=(("mp4 files","*.mp4"),("yuv files","*.yuv"),("avi files","*.avi")))

        
        #Creates a VideoCapture with the element selected
        self.first_video = cv2.VideoCapture(self.dir)

        
        #Checks if opened correctly
        if(self.first_video.isOpened() == False):
            print("Error opening video file")
        else:
            self.check1 = True
            self.checkLabel1.place(x=415,y=260) 

        if(self.check1 == True & self.check2 == True):
            self.buttonStart['state'] = 'normal'


    #Function open file 2
    def open_file2(self):
        self.dir = filedialog.askopenfilename(initialdir="/",title="Select Video",
         filetypes=(("mp4 files","*.mp4"),("yuv files","*.yuv"),("avi files","*.avi")))
        
        #Creates a VideoCapture with the element selected
        self.second_video = cv2.VideoCapture(self.dir)

        #Checks if opened correctly
        if(self.second_video.isOpened() == False):
            print("Error opening video file")
        else:
            self.check2 = True
            self.checkLabel2.place(x=415,y=325)

        #Checks if both videos are loaded
        if(self.check1 == True & self.check2 == True):
            self.buttonStart['state'] = 'normal'
        

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
    sigma_x_2 -= mu_x_2
    sigma_y_2 = cv.GaussianBlur(i2_2, (11, 11), 1.5)
    sigma_y_2 -= mu_y_2
    sigma_x_y = cv.GaussianBlur(i1_i2, (11, 11), 1.5)
    sigma_x_y -= mu_x_mu_y

    step1 = 2 * mu_x_mu_y + C1
    step2 = 2 * sigma_x_y + C2
    step3 = step1 * step2                    # t3 = ((2*mu_x_mu_y + C1).*(2*sigma_x2 + C2))
    step4 = mu_x_2 + mu_y_2 + C1
    step5 = sigma_x_2 + sigma_y_2 + C2
    step6 = step4 * step5                    # t1 =((mu_x_2 + mu_y_2 + C1).*(sigma_x_2 + sigma_y_2 + C2))

    ssim_map = cv.divide(step3, step6)    # ssim_map =  t3./t1;
    mssim = cv.mean(ssim_map)       # mssim = average of ssim map

    return mssim

if __name__=="__main__":
    QualiApp360()
