from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
from os import walk
import json

# oem 3 indica di utilizzare il motore di default
# psm 7 indica che il testo che stiamo leggendo è composto da una sola riga
# tessedit_char_whitelist indica la lisa dei caratteri ammessi
custom_oem_psm_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789 --tessdata-dir C:\\Users\\michi\\Desktop\\riconoscimento_numeri\\models\\'

def letterbox(image, new_shape=(320, 320), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """
    Ridimensiona l'immagine mantenendo il rapporto d'aspetto, aggiungendo bordi per ottenere new_shape.
    Restituisce:
      - l'immagine letterboxed
      - il fattore di scala usato
      - le dimensioni del padding (dw, dh)
    """
    shape = image.shape[:2]  # (altezza, larghezza)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calcola il fattore di scala per adattare l'immagine alla nuova dimensione
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # non ingrandire l'immagine se è già piccola
        r = min(r, 1.0)

    # Calcola la nuova dimensione senza padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    # Calcola il padding necessario
    dw = new_shape[1] - new_unpad[0]  # padding in larghezza
    dh = new_shape[0] - new_unpad[1]  # padding in altezza

    if auto:  # per ottenere multipli di 32 (come tipicamente richiesto da YOLO)
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0, 0
        new_unpad = new_shape
        r = new_shape[1] / shape[1]

    dw /= 2  # distribuisci il padding a sinistra e a destra
    dh /= 2  # distribuisci il padding in alto e in basso

    # Ridimensiona l'immagine se necessario
    if (shape[1], shape[0]) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    # Aggiungi i bordi
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (dw, dh)

def new_thres(img, h1 = 98, s1 = 0, v1 = 127, h2 = 155, s2 = 90, v2 = 255):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([h1, s1, v1])
    upper_white = np.array([h2, s2, v2])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    return mask

def calc_epsilon(image, e = 0.07110000000000001):
    contours, _ = cv2.findContours( image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    converted = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    cv2.drawContours(converted, contours, -1, (255, 0, 0), 3)

    angles = []
    approxed = []

    for contour in contours:
        epsilon = e * cv2.arcLength(contour, False)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x, y, w, h = cv2.boundingRect(approx)

        areaRect = cv2.minAreaRect(approx)
        angle = areaRect[2]

        if w > 30 and w < 100 and h > 20 and h < 60:
            cv2.drawContours(converted, [approx], -1, (0, 255, 0), 3)
            cut = image[y:y+h, x:x+w]

            approxed.append(cut)
            angles.append(- angle)
                
    return approxed, angles


cars = []

def recognize(source):

    image = cv2.imread(source)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    letterboxed_image, scale, (dw, dh) = letterbox(image, new_shape=(320, 320))
    result = model(letterboxed_image, imgsz=320).pop()

    if hasattr(result.boxes.data, "cpu"):
        recognitions = result.boxes.data.cpu().numpy()
    else:
        recognitions = result.boxes.data

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    expected_cars = input("number of cars:")
    expected_cars = int(expected_cars)

    numbers = []

    for rec in recognitions:
        x1, y1, x2, y2, conf, cls = rec
        x1_orig = int((x1 - dw) / scale)
        y1_orig = int((y1 - dh) / scale)
        x2_orig = int((x2 - dw) / scale)
        y2_orig = int((y2 - dh) / scale)

    
        # Recupera il nome della classe, se disponibile (il modello potrebbe avere l'attributo names)
        if hasattr(model, "names") and int(cls) in model.names:
            label = model.names[int(cls)]
        else:
            label = str(int(cls))
        
        if label == "car":
            car = image[y1_orig:y2_orig, x1_orig:x2_orig]
            
            cv2.imshow("box",car)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            expected = input("expected number:")
            
            
            expected = int(expected)
            thres = new_thres(car)
            
            boxes, angles = calc_epsilon(thres)
            found = []
            for box in boxes:
                text = pytesseract.image_to_string(box, config=custom_oem_psm_config)
                text = text.strip()

                if text != "":
                    number = int(text)
                    found.append(number)
                    if number == expected:
                        print("numero corretto", number)
            numbers.append({
                "bounds" : [x1_orig, y1_orig, x2_orig, y2_orig],
                "expected" : expected,
                "found" : found
            })
        cars.append({
            "image" : source,
            "expected_cars" : expected_cars,
            "cars_found" : len(boxes),
            "numbers" : numbers
        })
        with open(mypath + "\\cars.json", "w") as f:
            json.dump(cars, f, indent=4)

model = YOLO("C:\\Users\\michi\\Desktop\\riconoscimento_numeri\\notebooks\\yolov8m.onnx", task="detect")
mypath = "C:\\Users\\michi\\Desktop\\riconoscimento_numeri\\vids\\corti\\frames\\video1"

files = next(walk(mypath), (None, None, []))[2]

if __name__ == "__main__":
    bar_size = 100
    unit = len(files) / bar_size

    for i,fi in enumerate(files):

        percent = i * bar_size // len(files)

        print(f'{i}/{len(files)} { "#"* percent }{ "-" * (bar_size - percent)}')

        recognize(mypath + "\\" +fi)


