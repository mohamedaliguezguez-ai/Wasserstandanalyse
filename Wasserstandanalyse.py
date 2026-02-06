import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Wasserstand Detektor", layout="centered")

st.title("🥤 Wasserstand-Analyse Pro")
st.write("Erkenne den Füllstand in deiner Tasse oder deinem Glas.")

# --- SIDEBAR EINSTELLUNGEN ---
st.sidebar.header("Parameter")
sens_kreis = st.sidebar.slider("Kreis-Empfindlichkeit", 10, 100, 30, help="Niedriger = findet mehr Kreise")
min_dist = st.sidebar.slider("Min. Abstand Kreise", 10, 500, 100)
wasser_thresh = st.sidebar.slider("Wasser-Schwellenwert", 0, 255, 120, help="Regelt die Dunkelheit des Wassers")

# --- BILDQUELLE ---
# Wir nutzen camera_input für das Handy/Webcam-Gefühl
img_file = st.camera_input("Mache ein Foto von der Tasse")

if img_file is not None:
    # Bild konvertieren
    image = Image.open(img_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. KREIS ERKENNEN (Becher/Glas)
    # Weichzeichnen, um Rauschen zu mindern
    gray_blurred = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
        param1=50, param2=sens_kreis, minRadius=50, maxRadius=400
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Wir nehmen den ersten (meist besten) Kreis
        i = circles[0, 0]
        center = (i[0], i[1])
        radius = i[2]

        # Kreis zur Visualisierung zeichnen (Gelb für das Glas)
        cv2.circle(img_array, center, radius, (255, 255, 0), 5)
        
        # 2. WASSERSTAND ANALYSIEREN
        # Wir schauen uns nur den Bereich (ROI) innerhalb des Kreises an
        roi_x = max(0, i[0] - radius)
        roi_y = max(0, i[1] - radius)
        roi_w = radius * 2
        roi_h = radius * 2
        
        roi_gray = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        if roi_gray.size > 0:
            # Einfache Schwellenwert-Analyse für den Wasserstand
            # (Wasser ist oft dunkler oder bricht das Licht anders)
            _, binary = cv2.threshold(roi_gray, wasser_thresh, 255, cv2.THRESH_BINARY_INV)
            
            # Suche nach der obersten Kante der dunklen Fläche im ROI
            # Wir summieren die Zeilen auf
            row_sum = np.sum(binary, axis=1)
            water_indices = np.where(row_sum > (binary.shape[1] * 0.2)) # 20% der Zeile muss "Wasser" sein
            
            if len(water_indices[0]) > 0:
                top_water_rel = water_indices[0][0]
                top_water_abs = roi_y + top_water_rel
                
                # Wasserlinie zeichnen (Blau)
                cv2.line(img_array, (roi_x, top_water_abs), (roi_x + roi_w, top_water_abs), (0, 0, 255), 5)
                
                # Füllstand in % berechnen
                fuellstand = 100 - ((top_water_rel / roi_h) * 100)
                st.metric("Füllstand", f"{fuellstand:.1f} %")
            else:
                st.warning("Kein Wasser erkannt. Passe den Schwellenwert an.")

        st.image(img_array, caption="Analyse-Ergebnis", use_column_width=True)
    else:
        st.error("Kein Glas erkannt. Bitte positioniere die Tasse mittig.")