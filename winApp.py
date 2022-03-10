import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox,PhotoImage
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
from faceDetect import face_detector_hog, face_detector_cnn, face_detector_mtcnn, face_detector_haarcascades
import time
import json
import os
import shutil
from tkinter.filedialog import askopenfilenames, askopenfilename
from tkinter.ttk import Notebook
from tkinter import ttk 
from featureDataset import create_feature_ds
from featureClassification import euclidean_dist_classify, cosine_similarity_classify, svm_classify, train_svm_classification_model, knn_classify, train_knn_classification_model
from typing import Tuple
import numpy as np
import imutils
from preTrain import train_triplet_loss


names = []
fld_list = []
batch_size_chooser = [24, 48, 96, 288, 576]
input_chosser = ['Webcam', 'File']
classify_methods_chooser = ['Euclidean Distance', 'Cosine Similarity', 'SVM', 'KNN']
face_detect_methods_chooser = ['Dlib-HOG', 'Dlib-CNN', 'MTCNN', 'HaarCascades']
activate_face_detect_method = 'MTCNN'

class MainUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.read_user_json()
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Face Recognizer")
        self.resizable(False, False)
        self.geometry("500x250")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.activate_classify_method = 'Cosine Similarity'
        self.activate_name = None
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour, PageFive, PageSix):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
            frame = self.frames[page_name]
            frame.tkraise()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            self.destroy()

    def update_json_file(self):
        users_json = []
        for i, name in enumerate(names):
            user_json = {'name':name, 'fld_name':fld_list[i]}    
            users_json.append(user_json)
        data = {'users':users_json}
        with open(r'storage\something\users.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    def read_user_json(self):
        global names, fld_list
        with open(r'storage\something\users.json', encoding='utf-8') as json_file:
            data = json.load(json_file)
            users = data['users']
            for user in users:
                name = user['name']
                fld = user['fld_name']
                fld_list.append(fld)
                names.append(name)

    def delete_user(self):
        global names, fld_list
        activate_fld = fld_list[names.index(self.activate_name)]
        fld_list.remove(activate_fld)
        shutil.rmtree(os.path.join('storage\imageBase', activate_fld))
        names.remove(self.activate_name)
        self.update_json_file()
    
    def remove_all_samples(self):
        global names, fld_list
        activate_fld = fld_list[names.index(self.activate_name)]
        fld_dir = os.path.join('storage\imageBase', activate_fld)
        for f in os.listdir(fld_dir):
            os.remove(os.path.join(fld_dir, f))


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            self.controller = controller
            render = PhotoImage(file=r'storage\something\homepagepic.png')
            img = tk.Label(self, image=render)
            img.image = render
            img.grid(row=0, column=1, rowspan=5, sticky="nsew")
            label = tk.Label(self, text="        Home Page        ", font=self.controller.title_font,fg="#263942")
            label.grid(row=0, sticky="ew")
            button1 = tk.Button(self, text="   Face database  ", fg="#ffffff", bg="#263942",command=lambda:self.controller.show_frame("PageOne"))
            button2 = tk.Button(self, text="   Face Recognize  ", fg="#ffffff", bg="#263942",command=lambda:self.controller.show_frame("PageTwo"))
            button3 = tk.Button(self, text="   Train  ", fg="#ffffff", bg="#263942",command=lambda:self.controller.show_frame("PageThree"))
            button4 = tk.Button(self, text="   Quit  ", fg="#263942", bg="#ffffff", command=self.on_closing)
            button1.grid(row=1, column=0, ipady=3, ipadx=4)
            button2.grid(row=2, column=0, ipady=3, ipadx=1)
            button3.grid(row=3, column=0, ipady=3, ipadx=30)
            button4.grid(row=4, column=0, ipady=3, ipadx=31)
        
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            self.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Database", font=self.controller.title_font,fg="#263942", anchor="center")
        label.grid(column=0,row=0)
        button1 = tk.Button(self, text="  View Database ", fg="#ffffff", bg="#263942",command=lambda:self.controller.show_frame("PageFour"))
        button2 = tk.Button(self, text="Create new User", fg="#ffffff", bg="#263942",command=lambda:self.controller.show_frame("PageFive"))
        button3 = tk.Button(self, text="Back", bg="#ffffff", fg="#263942",command=lambda:self.controller.show_frame("StartPage"))
        button1.grid(row=1, ipady=3, ipadx=1)
        button2.grid(row=2, ipady=3, ipadx=1)
        button3.grid(row=4, ipady=3, ipadx=31)


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.webcam_ = None
        label = tk.Label(self, text="Face Recognitize", font=self.controller.title_font,fg="#263942", anchor="center")
        label.grid(row=0, column=0)
        tk.Label(self, text="Choose methods used in feature classification:").grid(row=1)
        self.method_classify_var = tk.StringVar(value=self.controller.activate_classify_method)
        self.methods_menu = tk.OptionMenu(self, self.method_classify_var, *classify_methods_chooser, command=self.change_method_classify)
        self.methods_menu.grid(row=2)
        self.face_reg_btn = tk.Button(self, text="Recognitize", command=lambda:self.recognitize())
        self.face_reg_btn.grid(row=3, column=0)
        self.buttoncanc = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942", command=lambda:self.cancel())
        self.buttoncanc.grid(row=4, column=0)

    def change_method_classify(self, event):
        self.controller.activate_classify_method = self.method_classify_var.get()
        # print(self.controller.activate_classify_method)

    def recognitize(self):
        self.webcam_ = webcam(self, controller=self.controller)
        self.webcam_.mode_indicate(mode="PageTwo")
    
    def cancel(self):
        if self.webcam_:
            self.webcam_.destroy()
        self.controller.show_frame("StartPage")


class PageThree(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.show_opt = False
        icon_1 = PhotoImage(file=r'storage\something\question.png')
        icon_2 = PhotoImage(file=r'storage\something\restore.png')
        icon_3 = PhotoImage(file=r'storage\something\graph.png')
        self.label = tk.Label(self, text="Training", font=self.controller.title_font,fg="#263942", anchor="center")
        self.label.grid(row=0, column=0)
        self.notebook = Notebook(self)
        self.notebook.grid(row=1, column=0)
        self.tab1 = tk.Frame(self.notebook)
        self.tab2 = tk.Frame(self.notebook)
        self.tab3 = tk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Train Model", compound=tk.TOP)
        self.notebook.add(self.tab2, text="Create Feature Dataset", compound=tk.TOP)
        self.notebook.add(self.tab3, text="Train Classify", compound=tk.TOP)
        self.back_btn = tk.Button(self, text="Back", bg="#ffffff", fg="#263942", command=lambda:self.cancel())
        self.back_btn.grid(row=3)
        # Train model panel
        self.train_btn = tk.Button(self.tab1, text="Train Model", fg="#ffffff", bg="#263942",command=lambda:self.train_model())
        self.train_btn.grid(row=0)
        self.adv_label = ttk.Label(self.tab1, text="Advanced")
        self.adv_label.bind("<Button-1>", lambda e:self.hide_show_opt())
        self.adv_label.grid(row=1)
        self.adv_frame = tk.Frame(self.tab1)
        self.adv_frame.grid_forget()
        tk.Label(self.adv_frame, text="Epochs:").grid(row=0, column=0)
        self.epochs_var = tk.DoubleVar(value=50)
        self.epochs_input = tk.Entry(self.adv_frame, textvariable = self.epochs_var)
        self.epochs_input.grid(row=0, column=1)
        tk.Label(self.adv_frame, text="Batch Size:").grid(row=1, column=0)
        self.batch_size_var = tk.DoubleVar(value=24)
        self.dropdown = tk.OptionMenu(self.adv_frame, self.batch_size_var, *batch_size_chooser)
        self.dropdown.config(bg="lightgrey")
        self.dropdown["menu"].config(bg="lightgrey")
        self.dropdown.grid(row=1, column=1)
        tk.Label(self.adv_frame, text="Learning Rate:").grid(row=2, column=0)
        self.lr_var = tk.DoubleVar(value=0.001)
        self.lr_input = tk.Entry(self.adv_frame, textvariable = self.lr_var)
        self.lr_input.grid(row=2, column=1)
        self.defaults_label = tk.Label(self.adv_frame, text="Reset defaults")
        self.defaults_label.bind("<Button-1>", lambda e:self.reset_defaults())       
        self.defaults_label.grid(row=3)
        self.underthehood_label = tk.Label(self.adv_frame, text="Under the hood")
        self.underthehood_label.bind("<Button-1>", lambda e:self.under_the_hood())     
        self.underthehood_label.grid(row=4)
        # Create feature dataset panel
        self.create_ds = tk.Button(self.tab2, text="Create Feature Dataset", command=lambda:self.create_f_ds())
        self.create_ds.grid(row=0)
        # Train classify panel
        tk.Label(self.tab3, text="Choose methods used in feature classification:").grid(row=0)
        self.method_classify_var = tk.StringVar(value=self.controller.activate_classify_method)
        self.methods_menu = tk.OptionMenu(self.tab3, self.method_classify_var, *classify_methods_chooser, command=self.train_classify_view)
        self.methods_menu.grid(row=1)
        self.train_classify_btn = tk.Button(self.tab3, text="train", command=lambda:self.train_classify())

    def train_classify_view(self, event):
        self.controller.activate_classify_method = self.method_classify_var.get()
        if self.controller.activate_classify_method in ['SVM', 'KNN', 'ANN']:
            self.train_classify_btn.grid(row=2)
        else:
            self.train_classify_btn.grid_forget()
    
    def train_classify(self):
        if self.controller.activate_classify_method == 'SVM':
            train_svm_classification_model()
        elif self.controller.activate_classify_method == 'KNN':
            train_knn_classification_model()

    def train_model(self):
        batch_size = self.batch_size_var.get()
        lr = self.lr_var.get()
        epochs = self.epochs_var.get()
        train_triplet_loss(batch_size_=batch_size,lr=lr,epochs=epochs)
    
    def hide_show_opt(self):
        if self.show_opt:
            self.show_opt = False
            self.adv_frame.grid_forget()
        else:
            self.show_opt = True
            self.adv_frame.grid(row=2)

    def reset_defaults(self):
        self.epochs_var.set(50)
        self.batch_size_var.set(24)
        self.lr_var.set(0.001)

    def under_the_hood(self):
        None

    def create_f_ds(self):
        create_feature_ds()
        messagebox.showinfo("Success","Feature dataset created successfull")

    def cancel(self):
        self.controller.show_frame("StartPage")  


class PageFour(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.webcam_ = None
        global names, fld_list
        self.add_img = tk.PhotoImage(file=r'storage\something\add.png')
        tk.Label(self, text="Select user", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, padx=10, pady=10)
        self.btn_back = tk.Button(self, text="Back", bg="#ffffff", fg="#263942", command=self.cancel)
        self.menuvar = tk.StringVar()
        self.dropdown = tk.OptionMenu(self, self.menuvar, *names)
        self.dropdown.config(bg="lightgrey")
        self.dropdown["menu"].config(bg="lightgrey")
        self.btn_view = tk.Button(self, text="View", command=self.view, fg="#ffffff", bg="#263942")
        self.dropdown.grid(row=0, column=1, ipadx=15, padx=10, pady=10)
        self.btn_view.grid(row=0, column=2, ipadx=5, ipady=4, pady=10)
        self.btn_back.grid(row=4, column=0)
        self.canvas = tk.Canvas(self, width=500, height=100)
        self.scrollbar = tk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e:self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.frame = tk.Frame(self.canvas)
        self.frame.bind("<Configure>", self.reset_scrollregion)
        self.canvas.create_window((0,0), window=self.frame, anchor='nw')
        self.edit_img = tk.PhotoImage(file=r'storage\something\edit.png')
        self.show_more_img = tk.PhotoImage(file=r'storage\something\show_more.png')
        self.classname_txt = tk.StringVar()
        self.classname = tk.Entry(self, textvariable=self.classname_txt, font='Helvetica 12 bold')
        self.btn_edit = tk.Button(self, image=self.edit_img, command=self.edit_classname)     
        self.btn_show_more = tk.Menubutton(self, image=self.show_more_img, relief="raised")
        self.btn_show_more.menu = tk.Menu(self.btn_show_more, tearoff = 0)
        self.btn_show_more['menu'] = self.btn_show_more.menu
        self.btn_show_more.menu.add_command(label="Delete User", command=lambda:self.delete_user())
        self.btn_show_more.menu.add_command(label="Remove All Samples", command=lambda:self.remove_all_samples())

    def reset_scrollregion(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def view(self):
        self.defaults()
        self.img_dir_list = []
        self.img_list = []
        if self.menuvar.get() == '':
            messagebox.showerror("ERROR", "Name cannot be 'None'")
            return
        global names, fld_list
        self.fld = fld_list[names.index(self.menuvar.get())]
        self.fld_path = os.path.join(r'storage\imageBase', self.fld)
        number = len(os.listdir(self.fld_path))
        photo_label = []
        k=None
        if number !=0:
            if number >= 8:
                for i, f in enumerate(os.listdir(self.fld_path)):
                    img_dir = os.path.join(self.fld_path,f)
                    self.img_dir_list.append(img_dir)
                    img = ImageTk.PhotoImage(Image.open(img_dir).resize((100, 100)))
                    self.img_list.append(img)
                    photo_label.append(tk.Label(self.frame, image=img))
            else:
                temp = []
                for i in range(number+1):
                    if i < number:
                        for f in os.listdir(self.fld_path):
                            temp.append(f)
                        img_dir = os.path.join(self.fld_path, temp[i])
                        self.img_dir_list.append(img_dir)
                        img = ImageTk.PhotoImage(Image.open(img_dir).resize((100, 100)))
                        self.img_list.append(img)
                        photo_label.append(tk.Label(self.frame, image=img))
                    else:
                        photo_label.append(tk.Label(self.frame, image=self.add_img))
                        k = i
        else:
            photo_label.append(tk.Label(self.frame, image=self.add_img))
            k = 0
        for m, n in enumerate(photo_label):
            if m == k:
                n.bind("<Button-1>", lambda e:self.add_a_image())
                n.grid(row=0, column=m)
            else:
                if m == 0:
                    n.bind("<Button-1>", lambda e:self.delete_a_image(0))
                if m == 1:
                    n.bind("<Button-1>", lambda e:self.delete_a_image(1))
                if m == 2:
                    n.bind("<Button-1>", lambda e:self.delete_a_image(2))
                if m == 3:
                    n.bind("<Button-1>", lambda e:self.delete_a_image(3))
                if m == 4:
                    n.bind("<Button-1>", lambda e:self.delete_a_image(4))
                if m == 5:
                    n.bind("<Button-1>", lambda e:self.delete_a_image(5))
                if m == 6:
                    n.bind("<Button-1>", lambda e:self.delete_a_image(6))
                if m == 7:
                    n.bind("<Button-1>", lambda e:self.delete_a_image(7))
                n.grid(row=0, column=m)
        self.canvas.grid(row=2, column=0, columnspan=4)
        self.scrollbar.grid(row=3, column=0, columnspan=4)
        self.controller.activate_name = self.menuvar.get()
        self.classname_txt.set(self.menuvar.get())
        self.classname.grid(row=1,column=0)
        self.btn_edit.grid(row=1,column=1)
        self.btn_show_more.grid(row=1,column=2)

    def delete_a_image(self, x):
        img_dir = self.img_dir_list[x]
        if messagebox.askyesno("Question", "Are you sure you want to delete this image?"):
            if os.path.exists(img_dir):
                os.remove(img_dir)
                if messagebox.showinfo("Success", "Image remove successfull"):
                    self.view()
            else:
                messagebox.showwarning("Warning", "Image does not exits")

    def add_a_image(self):
        self.webcam_ = webcam(self, controller=self.controller)
        self.webcam_.mode_indicate(mode="PageFour")

    def edit_classname(self):
        global names, fld_list
        if self.classname_txt.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif self.classname_txt.get() in names:
            messagebox.showerror("Error", "User already exists!")
            return
        elif len(self.classname_txt.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        activate_fld = fld_list[names.index(self.controller.activate_name)]
        fld_list.remove(activate_fld)
        shutil.rmtree(os.path.join('storage\imageBase', activate_fld))
        names.remove(self.controller.activate_name)
        names.append(self.classname_txt.get())
        fld_name = time.strftime("%d-%m-%y-%H-%M-%S")
        fld_list.append(fld_name)
        os.makedirs(os.path.join('storage\imageBase',fld_name))
        self.controller.activate_name = self.classname_txt.get()
        self.controller.update_json_file()
        self.controller.frames["PageFour"].refresh_names()
        messagebox.showinfo("Success", "Name change successfull")

    def delete_user(self):
        self.controller.delete_user()
        self.controller.activate_name = None
        self.defaults()
        self.refresh_names()
        if self.webcam_:
            self.webcam_.destroy()

    def remove_all_samples(self):
        self.controller.remove_all_samples()
        self.view()
    
    def refresh_names(self):
        global names
        self.menuvar.set('')
        self.dropdown['menu'].delete(0, 'end')
        for name in names:
            self.dropdown['menu'].add_command(label=name, command=tk._setit(self.menuvar, name))

    def cancel(self):
        if self.controller.activate_name:
            fld_name = fld_list[names.index(self.controller.activate_name)]
            fld_path = os.path.join(r'storage\imageBase', fld_name)
            if len(os.listdir(fld_path)) < 8:
                if messagebox.askokcancel("Warning", "Database for this user not enough 8 images.\nDo you want continues go back?"):
                    self.defaults()
                    self.menuvar.set('')
                    if self.webcam_:
                        self.webcam_.destroy()
                    self.controller.show_frame("PageOne")
            elif len(os.listdir(fld_path)) >= 8:
                self.defaults()
                self.menuvar.set('')
                if self.webcam_:
                    self.webcam_.destroy()
                self.controller.show_frame("PageOne")
        else:
            self.defaults()
            self.menuvar.set('')
            if self.webcam_:
                self.webcam_.destroy()
            self.controller.show_frame("PageOne")

    def defaults(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.canvas.grid_forget()
        self.scrollbar.grid_forget()
        self.classname.grid_forget()
        self.btn_edit.grid_forget()
        self.btn_show_more.grid_forget()


class PageFive(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text="Enter the name", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, pady=10, padx=5)
        self.user_name_txt = tk.StringVar()
        self.user_name = tk.Entry(self, textvariable=self.user_name_txt, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.user_name.grid(row=0, column=1, pady=10, padx=10)
        self.btn_back = tk.Button(self, text="Back", bg="#ffffff", fg="#263942", command=lambda:controller.show_frame("PageOne"))
        self.btn_create = tk.Button(self, text="Next", fg="#ffffff", bg="#263942", command=self.create_new)
        self.btn_back.grid(row=1, column=0, pady=10, ipadx=5, ipady=4)
        self.btn_create.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)

    def create_new(self):
        global names
        if self.user_name_txt.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif self.user_name_txt.get() in names:
            messagebox.showerror("Error", "User already exists!")
            return
        elif len(self.user_name_txt.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        name = self.user_name_txt.get()
        names.append(name)
        fld_name = time.strftime("%d-%m-%y-%H-%M-%S")
        fld_list.append(fld_name)
        os.makedirs(os.path.join('storage\imageBase',fld_name))
        self.controller.activate_name = name
        self.controller.update_json_file()
        self.controller.frames["PageFour"].refresh_names()
        self.controller.frames["PageSix"].update_classname()
        self.user_name_txt.set('')
        self.controller.show_frame("PageSix")


class PageSix(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.webcam_ = None
        self.controller = controller
        self.parent = parent
        self.add_img = tk.PhotoImage(file=r'storage\something\add.png')
        self.upload_img = tk.PhotoImage(file=r'storage\something\upload.png')
        self.webcam_img = tk.PhotoImage(file=r'storage\something\webcam.png')
        self.edit_img = tk.PhotoImage(file=r'storage\something\edit.png')
        self.show_more_img = tk.PhotoImage(file=r'storage\something\show_more.png')
        self.classname_txt = tk.StringVar()
        self.classname = tk.Entry(self, textvariable=self.classname_txt, font='Helvetica 12 bold')
        self.classname.grid(row=0,column=0, columnspan=2)
        self.btn_edit = tk.Button(self, image=self.edit_img, command=self.edit_classname)
        self.btn_edit.grid(row=0,column=2)
        self.btn_show_more = tk.Menubutton(self, image=self.show_more_img, relief="raised")
        self.btn_show_more.grid(row=0,column=3)
        self.btn_show_more.menu = tk.Menu(self.btn_show_more, tearoff = 0)
        self.btn_show_more['menu'] = self.btn_show_more.menu
        self.btn_show_more.menu.add_command(label="Delete User", command=lambda:self.delete_user())
        self.btn_show_more.menu.add_command(label="Remove All Samples", command=lambda:self.remove_all_samples())
        self.samples_label = tk.Label(self, text="Add image Samples:", fg="#263942")
        self.samples_label.grid(row=1,column=0)
        self.btn_webcam = tk.Button(self, text="Webcam", image=self.webcam_img, compound="left", command=lambda:self.webcam_view())
        self.btn_webcam.grid(row=2,column=0)
        self.btn_upload = tk.Button(self, text="Upload", image=self.upload_img, compound="left", command=lambda:self.upload_files())
        self.btn_upload.grid(row=2,column=1)
        self.buttoncanc = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942", command=lambda:self.cancel())
        self.buttoncanc.grid(row=5, column=0)
        self.canvas = tk.Canvas(self, width=500, height=100)
        self.scrollbar = tk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e:self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.frame = tk.Frame(self.canvas)
        self.frame.bind("<Configure>", self.reset_scrollregion)
        self.canvas.create_window((0,0), window=self.frame, anchor='nw')

    def reset_scrollregion(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def edit_classname(self):
        global names, fld_list
        if self.classname_txt.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif self.classname_txt.get() in names:
            messagebox.showerror("Error", "User already exists!")
            return
        elif len(self.classname_txt.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        activate_fld = fld_list[names.index(self.controller.activate_name)]
        fld_list.remove(activate_fld)
        shutil.rmtree(os.path.join('storage\imageBase', activate_fld))
        names.remove(self.controller.activate_name)
        names.append(self.classname_txt.get())
        fld_name = time.strftime("%d-%m-%y-%H-%M-%S")
        fld_list.append(fld_name)
        os.makedirs(os.path.join('storage\imageBase',fld_name))
        self.controller.activate_name = self.classname_txt.get()
        self.controller.update_json_file()
        self.controller.frames["PageFour"].refresh_names()
        messagebox.showinfo("Success", "Name change successfull")

    def delete_user(self):
        self.controller.delete_user()
        self.controller.activate_name = None
        self.controller.frames["PageFour"].refresh_names()
        self.webcam_.destroy()
        self.controller.show_frame("PageFive")
    
    def remove_all_samples(self):
        self.controller.remove_all_samples()
        self.view()

    def update_classname(self):
        self.classname_txt.set(self.controller.activate_name)       

    def webcam_view(self):
        self.webcam_ = webcam(self, controller=self.controller)
        self.webcam_.mode_indicate(mode="PageSix")
        self.view()
        # self.webcam_.grab_set()

    def view(self):
        self.defaults()
        self.img_dir_list = []
        self.img_list = []
        photo_label = []
        global names, fld_list
        self.fld = fld_list[names.index(self.controller.activate_name)]
        self.fld_path = os.path.join(r'storage\imageBase', self.fld)
        number = len(os.listdir(self.fld_path))
        self.samples_label['text'] = str(number) + ' image samples:'
        self.samples_label.grid(row=2,column=0)
        self.btn_webcam.grid(row=1,column=0)
        self.btn_upload.grid(row=1,column=1)
        if number !=0:
            for i, f in enumerate(os.listdir(self.fld_path)):
                img_dir = os.path.join(self.fld_path,f)
                self.img_dir_list.append(img_dir)
                img = ImageTk.PhotoImage(Image.open(img_dir).resize((100, 100)))
                self.img_list.append(img)
                photo_label.append(tk.Label(self.frame, image=img))
                if i == 0:
                    photo_label[i].bind("<Button-1>", lambda e:self.delete_a_image(0))
                if i == 1:
                    photo_label[i].bind("<Button-1>", lambda e:self.delete_a_image(1))
                if i == 2:
                    photo_label[i].bind("<Button-1>", lambda e:self.delete_a_image(2))
                if i == 3:
                    photo_label[i].bind("<Button-1>", lambda e:self.delete_a_image(3))
                if i == 4:
                    photo_label[i].bind("<Button-1>", lambda e:self.delete_a_image(4))
                if i == 5:
                    photo_label[i].bind("<Button-1>", lambda e:self.delete_a_image(5))
                if i == 6:
                    photo_label[i].bind("<Button-1>", lambda e:self.delete_a_image(6))
                if i == 7:
                    photo_label[i].bind("<Button-1>", lambda e:self.delete_a_image(7))
                photo_label[i].grid(row=0, column=i)
        self.canvas.grid(row=3, column=0, columnspan=4)
        self.scrollbar.grid(row=4, column=0, columnspan=4)

    def delete_a_image(self, x):
        img_dir = self.img_dir_list[x]
        if messagebox.askyesno("Question", "Are you sure you want to delete this image?"):
            if os.path.exists(img_dir):
                os.remove(img_dir)
                if messagebox.showinfo("Success", "Image remove successfull"):
                    self.view()
            else:
                messagebox.showwarning("Warning", "Image does not exits")

    def defaults(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.canvas.grid_forget()
        self.scrollbar.grid_forget()
        self.samples_label['text'] = 'Add image samples:'
        self.samples_label.grid(row=1,column=0)
        self.btn_webcam.grid(row=2,column=0)
        self.btn_upload.grid(row=2,column=1)
    
    def upload_files(self):
        file_path_list = askopenfilenames(filetypes=[('Image Files', '*jpeg'), ('Image Files', '*jpg'), ('Image Files', '*png')])
        if file_path_list != '':
            fld_name = fld_list[names.index(self.controller.activate_name)]
            fld_path = os.path.join(r'storage\imageBase', fld_name)
            count = 0
            for file_path in file_path_list:
                loaded_image = cv2.imread(file_path)
                frame, faces_img = self.get_face_detected(loaded_image)
                if faces_img:
                    for face_img in faces_img:
                        if len(os.listdir(fld_path)) >= 8:
                            if messagebox.showwarning("Warning", "Database for this user enough 8 images.\nPlease delete before add more image?"):
                                if messagebox.showinfo("Complete", "Add {} user face image(s)".format(count)):
                                    self.controller.frames["PageSix"].view()
                                    return
                        else:
                            path = os.path.join(fld_path, "img" + time.strftime("%d-%m-%y-%H-%M-%S") + ".jpg")
                            cv2.imwrite(path, face_img)
                            count += 1
            if messagebox.showinfo("Complete", "Add {} user face image(s)".format(count)):
                self.controller.frames["PageSix"].view()
    
    def get_face_detected(self, img):
        faces_img = []
        faces_location = []
        global activate_face_detect_method
        if activate_face_detect_method == "Dlib-HOG":
            faces_img, faces_location = face_detector_hog(img)
        elif activate_face_detect_method == "Dlib-CNN":
            faces_img, faces_location = face_detector_cnn(img)
        elif activate_face_detect_method == "MTCNN":
            faces_img, faces_location = face_detector_mtcnn(img)
        elif activate_face_detect_method == "HaarCascades":
            faces_img, faces_location = face_detector_haarcascades(img)
        if faces_img:
            for face_location in faces_location:
                (x, y, w, h) = face_location
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces_img)

    def cancel(self):
        fld_name = fld_list[names.index(self.controller.activate_name)]
        fld_path = os.path.join(r'storage\imageBase', fld_name)
        if len(os.listdir(fld_path)) < 8:
            if messagebox.askokcancel("Warning", "Database for this user not enough 8 images.\nDo you want continues go back?"):
                self.controller.show_frame("PageFive")
                self.defaults()
                if self.webcam_:
                    self.webcam_.destroy()
        elif len(os.listdir(fld_path)) >= 8:
            self.controller.show_frame("PageFive")
            self.defaults()
            if self.webcam_:
                self.webcam_.destroy()

class webcam(tk.Toplevel):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.title('Webcam')
        self.mode = None
        self.photo_gray = None      
        self.webcam_img = tk.PhotoImage(file=r'storage\something\webcam.png')
        self.add_image_img = tk.PhotoImage(file=r'storage\something\add_image.png')
        add_image_img_gray_ = cv2.cvtColor(cv2.imread(r'storage\something\add_image.png'), cv2.COLOR_BGR2GRAY)
        add_image_img_gray_arr = Image.fromarray(add_image_img_gray_)
        self.add_image_img_gray = ImageTk.PhotoImage(add_image_img_gray_arr)
        self.loaded_photo = self.add_image_img
        self.loaded_photo_gray = self.add_image_img_gray
        self.iconphoto(False, self.webcam_img)
        self.video_source = 0
        self.vid = video_capture(self.video_source)
        self.on_img = tk.PhotoImage(file=r'storage\something\toggle_on.png')
        self.off_img = tk.PhotoImage(file=r'storage\something\toggle_off.png')
        self.input = tk.Label(self, text="Input")       
        self.is_on = True
        self.status = tk.Label(self, text="On")
        self.toggle_btn = tk.Button(self)
        self.toggle_btn.configure(image=self.on_img, command=lambda:self.switch())  
        self.menuvar = tk.StringVar()
        self.menuvar.set('Webcam')
        self.canvas = tk.Canvas(self,width=self.vid.width,height=self.vid.height,bg='black')
        self.label = tk.Label(self,image=self.add_image_img,width=self.vid.width,height=self.vid.height)
        self.label.bind("<Button-1>", lambda e:self.upload_file())
        self.btn_snapshot = tk.Button(self,text='Snapshot',width=30,bg='goldenrod2',activebackground='red',command=self.snapshot)
        self.dropdown = tk.OptionMenu(self, self.menuvar, *input_chosser)
        self.dropdown.config(bg="lightgrey")
        self.dropdown["menu"].config(bg="lightgrey") 
        tk.Label(self,text="Choose method to detect face:").grid(row=1)
        global activate_face_detect_method
        self.detect_method_var = tk.StringVar(value=activate_face_detect_method)
        self.detect_method_menu = tk.OptionMenu(self, self.detect_method_var, *face_detect_methods_chooser, command=self.detect_method)
        self.detect_method_menu.grid(row=2)
        self.update()

    def detect_method(self, event):
        global activate_face_detect_method
        activate_face_detect_method = self.detect_method_var.get()

    def mode_indicate(self, mode):
        self.mode = mode
        self.menuvar.set('Webcam')
        if mode == "PageFour":
            self.input.grid(row=0, column=0)
            self.toggle_btn.grid(row=0, column=2)
            self.status.grid(row=0, column=3)
            self.dropdown.grid(row=0, column=4)
            self.btn_snapshot.grid(row=4, column=0, columnspan=5)
        if mode == "PageSix":
            self.input.grid(row=0, column=0)
            self.toggle_btn.grid(row=0, column=2)
            self.status.grid(row=0, column=3)
            self.dropdown.grid_forget()
            self.btn_snapshot.grid(row=4, column=0, columnspan=5)
        if mode == "PageTwo":
            self.input.grid(row=0, column=0)
            self.toggle_btn.grid(row=0, column=2)
            self.status.grid(row=0, column=3)
            self.dropdown.grid(row=0, column=4)
            self.btn_snapshot.grid_forget()

    def switch(self):
        if self.is_on:
            self.toggle_btn.config(image = self.off_img)
            self.status['text'] = "Off"
            self.is_on = False
        else:
            self.toggle_btn.config(image = self.on_img)
            self.status['text'] = "On"
            self.is_on = True

    def input_switch(self):
        if self.menuvar.get() == "File":
            self.canvas.grid_forget()
            self.label.grid(row=3, column=0, columnspan=5)
            self.btn_snapshot.grid_forget()
        elif self.menuvar.get() == "Webcam":
            self.label.grid_forget()
            self.canvas.grid(row=3, column=0, columnspan=5)
            if self.mode != "PageTwo":
                self.btn_snapshot.grid(row=4, column=0, columnspan=5)

    def upload_file(self):
        self.label.bind("<Button-1>", lambda e:self.disable())
        if self.mode == "PageFour":
            file_path = askopenfilename(filetypes=[('Image Files', '*jpeg'), ('Image Files', '*jpg'), ('Image Files', '*png')])
            if file_path != '':
                fld_name = fld_list[names.index(self.controller.activate_name)]
                fld_path = os.path.join(r'storage\imageBase', fld_name)
                loaded_image = cv2.imread(file_path)
                frame, faces_img = self.get_face_detected(loaded_image)
                if faces_img:
                    if len(os.listdir(fld_path)) < 8:
                        for face_img in faces_img:
                            path = os.path.join(fld_path, "img" + time.strftime("%d-%m-%y-%H-%M-%S") + ".jpg")
                            cv2.imwrite(path, face_img)
                            self.controller.frames["PageFour"].view()
                        if messagebox.showinfo("Success", "Save image successfull"):
                            self.label.bind("<Button-1>", lambda e:self.upload_file())
                            self.destroy()
                    else:
                        if messagebox.showwarning("Warning", "Database for this user enough 8 images.\nPlease delete before add more image"):
                            self.label.bind("<Button-1>", lambda e:self.upload_file())
                            self.destroy()
                else:
                    if messagebox.showerror("Error", "No face detected"):
                        self.label.bind("<Button-1>", lambda e:self.upload_file())

    def get_face_detected(self,img):
        faces_img = []
        faces_location = []
        global activate_face_detect_method
        if activate_face_detect_method == "Dlib-HOG":
            faces_img, faces_location = face_detector_hog(img)
        elif activate_face_detect_method == "Dlib-CNN":
            faces_img, faces_location = face_detector_cnn(img)
        elif activate_face_detect_method == "MTCNN":
            faces_img, faces_location = face_detector_mtcnn(img)
        elif activate_face_detect_method == "HaarCascades":
            faces_img, faces_location = face_detector_haarcascades(img)
        if faces_img:
            for face_location in faces_location:
                (x, y, w, h) = face_location
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), faces_img)

    def snapshot(self):
        self.btn_snapshot.configure(command=self.disable)
        if self.mode == "PageSix":
            is_true, frame, faces_image = self.vid.get_face_detected()
            if is_true:
                fld_name = fld_list[names.index(self.controller.activate_name)]
                fld_path = os.path.join(r'storage\imageBase', fld_name)
                if faces_image:
                    if len(os.listdir(fld_path)) < 8:
                        path = os.path.join(fld_path, "img" + time.strftime("%d-%m-%y-%H-%M-%S") + ".jpg")
                        cv2.imwrite(path, faces_image[0])
                        if messagebox.showinfo("Success", "Save image successfull"):
                            self.controller.frames["PageSix"].view()
                    else:
                        if messagebox.showwarning("Warning", "Database for this user enough 8 images.\nPlease delete before add more image"):
                            self.btn_snapshot.configure(command=self.snapshot)
                            self.destroy()
                else:
                    if messagebox.showerror("Error", "No face detected"):
                        self.btn_snapshot.configure(command=self.snapshot)
        elif self.mode == "PageFour":
            is_true, frame, faces_image = self.vid.get_face_detected()
            if is_true:
                fld_name = fld_list[names.index(self.controller.activate_name)]
                fld_path = os.path.join(r'storage\imageBase', fld_name)
                if faces_image:
                    path = os.path.join(fld_path, "img" + time.strftime("%d-%m-%y-%H-%M-%S") + ".jpg")
                    cv2.imwrite(path, faces_image[0])
                    # messagebox.showinfo("Success", "Save image successfull")
                    self.controller.frames["PageFour"].view()
                    self.btn_snapshot.configure(command=self.snapshot)
                    self.destroy()
                else:
                    self.btn_snapshot.configure(command=self.snapshot)
                    if messagebox.showwarning("Warning", "No face detected"):
                        self.btn_snapshot.configure(command=self.snapshot)

    def update(self):
        if self.mode == "PageTwo":
            if self.is_on:
                if self.menuvar.get() == "Webcam":
                    is_true, frame, faces = self.vid.get_face_detected()
                    if is_true:
                        if faces:
                            # for face in faces:
                            #     resize_face = cv2.resize(face, (224,224))
                            #     user_name = self.get_face_info(resize_face)
                            #     if user_name:
                            #         frame = cv2_img_add_text(frame,user_name,(0,5),(0,0,255))
                            resize_face = cv2.resize(faces[0], (224,224))
                            user_name = self.get_face_info(resize_face)
                            if user_name:
                                frame = cv2_img_add_text(frame,user_name,(0,5),(0,0,255))
                        self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
                        self.photo_gray = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                        self.canvas.create_image(0,0,image=self.photo,anchor='nw')
                elif self.menuvar.get() == "File":
                    self.label.configure(image=self.loaded_photo)
                    self.label.bind("<Button-1>", lambda e:self.load_image())
            else:
                if self.menuvar.get() == "Webcam":
                    if self.photo_gray:
                        self.canvas.create_image(0,0,image=self.photo_gray,anchor='nw')
                elif self.menuvar.get() == "File":
                    if self.loaded_photo_gray:
                        self.label.configure(image=self.loaded_photo_gray)
                    self.label.bind("<Button-1>", lambda e:self.disable())
        else:
            if self.is_on:
                is_true, frame = self.vid.get_frame()
                self.label.bind("<Button-1>", lambda e:self.upload_file())
                self.btn_snapshot.configure(command=self.snapshot)
                if is_true:
                    self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
                    self.photo_gray = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                    self.canvas.create_image(0,0,image=self.photo,anchor='nw')
            else:
                self.label.configure(image=self.add_image_img_gray)
                if self.photo_gray:
                    self.canvas.create_image(0,0,image=self.photo_gray,anchor='nw')
                self.label.bind("<Button-1>", lambda e:self.disable())
        self.input_switch()
        self.after(15, self.update)
        
    def disable(self):
        None

    def load_image(self):
        file_path = askopenfilename(filetypes=[('Image Files', '*jpeg'), ('Image Files', '*jpg'), ('Image Files', '*png')])
        if file_path != '':
            img = cv2.imread(file_path)
            height = img.shape[0]
            width = img.shape[1]
            channels = img.shape[2]
            if height > 480:
                resize_img = imutils.resize(img, height=480)
            elif width > 640:
                resize_img = imutils.resize(img, width=640)
            else:
                resize_img = img
            frame, faces = self.get_face_detected(img)
            if faces:
                resize_face = cv2.resize(faces[0], (224,224))
                user_name = self.get_face_info(resize_face)
                if user_name:
                    show = cv2_img_add_text(resize_img,user_name,(0,5),(0,0,255))
                    self.loaded_photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(show, cv2.COLOR_BGR2RGB)))
                    self.loaded_photo_gray = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)))
            else:
                self.loaded_photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)))
                messagebox.showerror("Error","No face detected")

    def get_face_info(self, face_pixels):
        label = None
        show = None
        # print(self.controller.activate_classify_method)
        if self.controller.activate_classify_method == "Euclidean Distance":
            label, prob = euclidean_dist_classify(face_pixels)
            show = 'User : %s \n Dist: %.3f' % (label, prob)
            # if min_dist > 0.45:
            #     label = 'Unknown'
        elif self.controller.activate_classify_method == "Cosine Similarity":
            label, prob = cosine_similarity_classify(face_pixels)
            show = 'User : %s \n Prob: %.3f' % (label, prob)
        elif self.controller.activate_classify_method == "SVM":
            label, prob = svm_classify(face_pixels)
            show = 'User : %s \n Prob: %.3f' % (label, prob)
        elif self.controller.activate_classify_method == "KNN":
            label = knn_classify(face_pixels)
            show = 'User : %s' % (label)
        return show

def cv2_img_add_text(img, text, left_corner: Tuple[int, int], text_rgb_color=(255, 0, 0), text_size=24, font=r'storage\something\arial.ttc', **option):
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_text = ImageFont.truetype(font=font, size=text_size, encoding=option.get('encoding', 'utf-8'))
    draw.text(left_corner, text, text_rgb_color, font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get('replace'):
        img[:] = cv2_img[:]
        return None
    return cv2_img


# from PIL import ImageGrab


# class video_capture:
#     def __init__(self, video_source = 0):
#         self.width = 640
#         self.height = 360
    
#     def get_frame(self):
#         if True:
#             is_true, frame = True, cv2.cvtColor(np.asarray(ImageGrab.grab().resize((640,360))), cv2.COLOR_BGR2RGB)
#             if is_true:
#                 return (is_true, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             else:
#                 return (is_true, None)
#         else:
#             return (is_true, None)

#     def get_face_detected(self):
#         if True:
#             is_true, frame = True, cv2.cvtColor(np.asarray(ImageGrab.grab()), cv2.COLOR_BGR2RGB)
#             if is_true:
#                 faces_img = []
#                 faces_location = []
#                 global activate_face_detect_method
#                 if activate_face_detect_method == "Dlib-HOG":
#                     faces_img, faces_location = face_detector_hog(frame)
#                 elif activate_face_detect_method == "Dlib-CNN":
#                     faces_img, faces_location = face_detector_cnn(frame)
#                 elif activate_face_detect_method == "MTCNN":
#                     faces_img, faces_location = face_detector_mtcnn(frame)
#                 elif activate_face_detect_method == "HaarCascades":
#                     faces_img, faces_location = face_detector_haarcascades(frame)
#                 if faces_img:
#                     for face_location in faces_location:
#                         (x, y, w, h) = face_location
#                         cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
#                 return (is_true, cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),(640,360)), faces_img)
#             else:
#                 return (is_true, None, None)
#         else:
#             return (is_true, None, None)
    
#     def __del__(self):
#         if self.vid.isOpened():
#             self.vid.release()


class video_capture:
    def __init__(self, video_source = 0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open this camera \n Select another video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    def get_frame(self):
        if self.vid.isOpened():
            is_true, frame = self.vid.read()
            if is_true:
                return (is_true, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (is_true, None)
        else:
            return (is_true, None)

    def get_face_detected(self):
        if self.vid.isOpened():
            is_true, frame = self.vid.read()
            if is_true:
                faces_img = []
                faces_location = []
                global activate_face_detect_method
                if activate_face_detect_method == "Dlib-HOG":
                    faces_img, faces_location = face_detector_hog(frame)
                elif activate_face_detect_method == "Dlib-CNN":
                    faces_img, faces_location = face_detector_cnn(frame)
                elif activate_face_detect_method == "MTCNN":
                    faces_img, faces_location = face_detector_mtcnn(frame)
                elif activate_face_detect_method == "HaarCascades":
                    faces_img, faces_location = face_detector_haarcascades(frame)
                if faces_img:
                    for face_location in faces_location:
                        (x, y, w, h) = face_location
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                return (is_true, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), faces_img)
            else:
                return (is_true, None, None)
        else:
            return (is_true, None, None)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


app = MainUI()
app.iconphoto(False, tk.PhotoImage(file=r'storage\something\facerecog.png'))
app.mainloop()
