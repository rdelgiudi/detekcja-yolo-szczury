# Śledzenie szczurów poprzez detekcję z wykorzystaniem YOLOv3 i OpenCV
Implementacja detektora YOLOv3 wraz z wytrenowanym modelem.

## Instalacja

Program testowany w Pythonie 3.8.5

### Wymagane biblioteki
```Linux
pip install -r listabibliotekpython.txt
```

### Pełne biblioteki (wymagane do trybu Cython. Uwaga! Potrzebny jest też kompilator C)
```Linux
pip install -r listabibliotekpythonfull.txt
```
Pełne możliwości programu można uzyskać tylko dla skompilowanej wersji OpenCV w wersji 4.4.0.

Więcej informacji: [Build OpenCV 4.4.0 with CUDA (GPU) Support on Windows 10 (Without Tears)](https://haroonshakeel.medium.com/build-opencv-4-4-0-with-cuda-gpu-support-on-windows-10-without-tears-aa85d470bcd0)

## Użycie
```Linux
python start.py -v VIDEO [-h] [-o OUTPUT] [-f] [-c CONFIDENCE] [-t THRESHOLD] 
                [-d] [-m MAXLOSS] [-cy] [-l] [-cc]

wymagane argumenty:
  -v VIDEO, --video VIDEO                 ścieżka pliku wideo

opcjonalne argumenty:
  -h, --help                              wyświetl tą wiadomość pomocy i wyjdź
  -o OUTPUT, --output OUTPUT              plik wynikowy śledzenia
  -f, --fileonly                          operacje tylko na pliku, brak podglądu
  -c CONFIDENCE, --confidence CONFIDENCE  minimalna pewność algorytmu
  -t THRESHOLD, --threshold THRESHOLD     granica non maximum supression
  -d, --drawpaths                         rysuj ścieżki śledzonych obiektów
  -m MAXLOSS, --maxloss MAXLOSS           maksymalna ilość klatek, 
                                          na którą może zniknąć obiekt,
                                          aby jego ścieżka nie została
                                          wyczyszczona (ta wartość +5)
  -cy, --cythonmode                       tryb cython dla poprawy wydajności 
                                          (wymaga biblioteki Cython)
  -l, --logging                           zapisz raport ze śledzenia w pliku tekstowym
  -cc, --calccross                        oblicz ilość przecięć dróg
  ```
  
  ## Opis programu
  Jest to moja implementacja detektora YOLOv3, do użytku do obserwacji szczurów. Wraz ze standardową procedurą inicjalizacji oraz ułożenia wyników detekcji znajduje się tu także prymitywny system ID oraz rysowania ścieżek.
  
  ### Użyte źródła
  [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
  
  [Train YOLO to detect a custom object (online with free GPU)](https://pysource.com/2020/04/02/train-yolo-to-detect-a-custom-object-online-with-free-gpu/)
  
  [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
