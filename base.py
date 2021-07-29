from tkinter import *
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo


window = Tk()

window.title("QualiApp360")
logo = PhotoImage(file='LOGO48_full.png')
window.iconphoto(False, logo)
window.geometry('1080x600')

def select_file():
    filetypes = (
        ('mp4 files', '*.mp4'),
        ('yuv files', '*.yuv')
    )
    filename = fd.askopenfilename(
        title = "Open File",
        initialdir = '/',
        filetypes = filetypes
    )
"""
    showinfo(
        title='Selected File'
        #message=filename
    )
""" 

open_button = Button(
    window,
    text = 'Open File',
    command = select_file
)

open_button.grid(column=2, row=0)

open_button.pack(expand=True)

window.mainloop()
