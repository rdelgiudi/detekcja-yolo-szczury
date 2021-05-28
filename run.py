import numpy as np
import argparse
import readline
import os.path
import cv2
import progressbar


ap = argparse.ArgumentParser(add_help=False)
requiredNamed = ap.add_argument_group('wymagane argumenty')
optionalNamed = ap.add_argument_group('opcjonalne argumenty')
#(args, rest) = ap.parse_known_args()

requiredNamed.add_argument("-v", "--video", type = str,
    help = "ścieżka pliku wideo", required=True)
optionalNamed.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
    help='wyświetl tą wiadomość pomocy i wyjdź')
optionalNamed.add_argument("-o", "--output", type = str, default = None,
    help = "plik wynikowy śledzenia")
optionalNamed.add_argument("-f", "--fileonly", action="store_true",
    help = "operacje tylko na pliku, brak podglądu")
optionalNamed.add_argument("-c", "--confidence", type = float, default=0.5,
    help = "minimalna pewność algorytmu")
optionalNamed.add_argument("-t", "--threshold", type = float, default=0.3,
    help = "granica non maximum supression")
optionalNamed.add_argument("-d", "--drawpaths", action="store_true",
    help = "rysuj ścieżki śledzonych obiektów")
optionalNamed.add_argument("-m", "--maxloss", type = int, default=1000,
    help = "maksymalna ilość klatek, na którą może zniknąć obiekt, aby jego ścieżka nie została wyczyszczona (ta wartość +5)")
optionalNamed.add_argument("-cy", "--cythonmode", action="store_true",
    help = "tryb cython dla poprawy wydajności (wymaga biblioteki Cython)")
optionalNamed.add_argument("-l", "--logging", action="store_true",
    help = "zapisz raport ze śledzenia w pliku tekstowym")
optionalNamed.add_argument("-cc", "--calccross", action="store_true",
    help = "oblicz ilość przecięć dróg")
optionalNamed.add_argument("-p", "--pixelspercm", type = float, default=0,
    help = "ilość pikseli składających się na jeden cm")

args = ap.parse_args()

if not os.path.isfile(args.video):
    print("[BŁĄD] Podana nazwa pliku nie istnieje!")
    exit(0)

if args.cythonmode:
    import pyximport; pyximport.install(language_level=3, setup_args={'include_dirs': np.get_include()})
    import yolodetect
else:
    import yolodetectog as yolodetect

#Sledzenie obrazu domyślnego jeśli brak filmu
#jeśli jest uruchamiamy Sledzenie z podanego filmu
vs = cv2.VideoCapture(args.video)
bar = progressbar.ProgressBar()

yolodetect.startDetect(args.video ,vs ,args.confidence, args.threshold, args.output, args.fileonly, args.drawpaths, 
                        args.maxloss, args.logging, args.calccross, args.pixelspercm, False , bar)

