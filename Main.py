# Eigene Dateien
import ModelLoader
import Erkennen
# Von Python
import os
from tkinter import filedialog, messagebox  # für das GUI
import cv2  # Für das Verabeiten des Bildes
from tkinter import *  # für das GUI
from PIL import Image, ImageDraw  # Für das Zeichen des Bildes in einer Canvas/Leinwand

# Das trainierte Model wird geladen
model = ModelLoader.loadmodel('EMNISTModel.pt', 'cpu')

# Parameter der Fenster
width = 400
height = 400

# Das Bild
image = Image.new('RGB', (width, height), (0, 0, 0))
##image2 = Image.open("img2.jpg")
draw = ImageDraw.Draw(image)
path = 'img.jpg'

# Main Window
master = Tk()
master.title("RECONNAISSANCE DE L'ÉCRITURE MANUSCRITE")
master.resizable(width=False, height=False)
master.geometry('1000x800+400+120')
##master.tk.call('wm', 'iconphoto', master._w, PhotoImage(file='handschrift.png'))  # Main window iconbild
master.configure(background="#7f8c8d")


def lesen():
    """letze Bild wird gespeichert und erkannt"""
    global andere  # top 5 Prognosen
    image.save(path)  # das Bild wird zuerst gespeischert
    # Das Zeichnen wird erkannt
    top1, top5 = Erkennen.erkennen(path, model)  # aus Class Erkennen
    andere = []
    mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
    prognose_1.configure(text=mapping[top1])  # prognose_1 ist ein Label zur Prognose-Anzeige
    for val in top5[1:5]:
        andere.append(mapping[val])
        prognose_2['text'] = mapping[top5[1]]  # prognose_2 bis prognose_5 sind Buttons zur Prognose-Anzeige
        prognose_3['text'] = mapping[top5[2]]
        prognose_4['text'] = mapping[top5[3]]
        prognose_5['text'] = mapping[top5[4]]
    Leinwand.delete('all')  # Leinwand räumen
    draw.rectangle((0, 0, 400, 400), fill='black')  # das Bild löschen
    label_Erkannt.grid(row=10, column=0, rowspan=2, columnspan=3)
    # Zeigt was gezeichnnet wird
    bild = cv2.imread(path)
    bild = cv2.resize(bild, (100, 100))
    cv2.imshow('Bild', bild)
    cv2.waitKey(10)
    cv2.resizeWindow("Bild", 200, 200)


def paint(event):
    """Zeichen"""
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    Leinwand.create_oval(x1, y1, x2, y2, fill='black')  # in dem canvas zeichnen
    draw.ellipse([x1, y1, x2, y2], fill='white', outline='white')  # das Bild zeichnen


# Canvas Leinwand
Leinwand = Canvas(master, width=width, height=height, bg='#f0b27a', cursor="pencil")
Leinwand.bind("<B1-Motion>", paint)
Leinwand.grid(row=3, column=0, rowspan=7, columnspan=3)


def DataEingeben():
    '''eingene Handchrift eingeben'''
    try:
        global andere
        andere = []
        path = filedialog.askopenfilename()
        top1, top5 = Erkennen.erkennen(path=path, model=model)
        mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
        prognose_1.configure(text=mapping[top1])
        for val in top5[1:]:
            andere.append(mapping[val])
        prognose_2['text'] = mapping[top5[1]]
        prognose_3['text'] = mapping[top5[2]]
        prognose_4['text'] = mapping[top5[3]]
        prognose_5['text'] = mapping[top5[4]]
        label_Erkannt.grid(row=10, column=0, rowspan=2, columnspan=3)
        cv2.destroyAllWindows()
    except TypeError and cv2.error as T:
        # Fehler box
        messagebox.showerror("Fehler", T)


def räumen():
    ''' Leinwand leer machen und das Bild Löschen '''
    cv2.destroyAllWindows()
    Leinwand.delete('all')
    draw.rectangle((0, 0, 400, 400), fill='black')
    prognose_1.configure(text="")
    label_Erkannt.grid_forget()


# Einführung text ganz oben dem Mainwindow
label_hier_schreiben = Label(master, width=30, height=2, bg="#7f8c8d", text='Écrivez ici',
                             font=("Arial Bold", 25))
label_hier_schreiben.grid(row=0, column=0, rowspan=2, columnspan=3)

# Label Erkannt
label_Erkannt = Label(master, width=30, height=2, text="Reconnue", bg="#7f8c8d", font=("Arial Bold", 25))

# Label Erste Prognose
prognose_1 = Label(master, width=10, height=1, text="", bg="#7f8c8d", font=("Arial Bold", 50),
                   fg='red')
prognose_1.grid(row=12, column=0, rowspan=2, columnspan=3)

# zweite top Prognose
prognose_2 = Button(text='', height=1, width=5,
                    bg="#7f8c8d", fg='#350b02', font=("Arial Bold", 25), highlightbackground='lightblue2')
prognose_2.grid(row=4, column=3)

# dritte top Prognose
prognose_3 = Button(text='', height=1, width=5, bg="#7f8c8d",
                    fg='#350b02', font=("Arial Bold", 25), highlightbackground='lightblue2')
prognose_3.grid(row=5, column=3)

# vierte top Prognose
prognose_4 = Button(text='', height=1, width=5, bg="#7f8c8d",
                    fg='#350b02', font=("Arial Bold", 25), highlightbackground='lightblue2')
prognose_4.grid(row=4, column=4)

# quinte top Prognose
prognose_5 = Button(text='', height=1, width=5, bg="#7f8c8d",
                    fg='#350b02', font=("Arial Bold", 25), highlightbackground='lightblue2')
prognose_5.grid(row=5, column=4)

# Label Andere Erkennugen
andere_Erkennugen = Label(master, width=20, height=2, text='Autres détections', bg="#7f8c8d", font=("Arial Bold", 25))
andere_Erkennugen.grid(row=3, column=3, columnspan=2)

# Lesen button
button_lesen = Button(text='lire', height=1, width=8, command=lesen, highlightbackground='#e1243e',
                      bg="burlywood", relief=RAISED, bd=5, cursor="hand2", font=("Arial Bold", 22))
button_lesen.grid(row=10, column=3)

# Eingabe button
button_eingeben = Button(text="Entrez une \nécriture", height=2, width=9, command=DataEingeben,
                         highlightbackground='lightblue2', bg="beige", bd=5, cursor="hand2", font=("Arial Bold", 20))
button_eingeben.grid(row=10, column=4, rowspan=1, columnspan=5)

# Leinwand räume Button
button_räumen = Button(text="effacer", height=1, width=8, command=räumen,
                       highlightbackground='lightblue2', bg="beige", bd=5, cursor="hand2", font=("Arial Bold", 20))
button_räumen.grid(row=14, column=3, rowspan=1, columnspan=5)

# Den Autor anzeigen
label_Autor = Label(master, width=20, height=3, text='By Armel Franck\n# Jan 2020', bg="#7f8c8d",
                    font=("Arial Bold", 20))
label_Autor.grid(row=9, column=3, columnspan=2)

# Main loop
mainloop()
