import cv2
import numpy as np
from torchvision import transforms

"""Verarbeitet  das Eingangsbild und speichert es in das Modell"""


def erkennen(path, model):
    """path --> Pfad zum Bild.
       model --> das trainierte Modell.
       Return --> Top1-Ergebnis und Top5-Ergebnis.
    """
    try:
        # Eingangsdaten verarbeiten
        input_bild = cv2.imread(path)
        input_bild = cv2.resize(input_bild, (28, 28))
        input_bild = cv2.cvtColor(input_bild, cv2.COLOR_BGR2GRAY)  # zu gray
        input_bild = np.fliplr(input_bild)  # flip links/rechts
        input_bild = np.rot90(input_bild)  # um 90 Grad drehen
        trans = transforms.ToTensor()  # to tensor transformieren
        input_bild = trans(input_bild)
        input_bild.unsqueeze_(0)  # eine Dimension hinzufügen

        # Prognose methode
        output = model(input_bild)
        pred_1 = output.argmax(dim=1, keepdim=True)
        _, pred_5 = output.topk(5, dim=1)
        return pred_1.item(), pred_5.tolist()[0]
    except TypeError as TE:
        print(TE)
    except NameError as NE:
        print(NE)
