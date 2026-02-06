import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Wasserstand Detektor", layout="centered")
st.title("🥤 Wasserstand-Analyse Pro")

# Sidebar für die Regler
st.sidebar.header("Parameter")
wasser_thresh = st.sidebar.slider("Wasser-Schwellenwert", 0, 255, 120)

img_file = st.camera_input("Foto machen")

if img_file is not None:
    image = Image.open(img_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Kreis-Erkennung (Becher)
    gray_blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HOUGH_GRADIENT # Initialisierung
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=50, maxRadius=400)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        i = circles[0, 0]
        center, radius = (i[0], i[1]), i[2]

        # --- MASKE ERSTELLEN ---
        # Wir schauen NUR in den gelben Kreis
        mask = np.zeros_like(gray)
        cv2.circle(mask, center, radius, 255, -1)
        masked_gray = cv2.bitwise_and(gray, mask)

        # Bereich (ROI) ausschneiden
        y_s, y_e = max(0, i[1]-radius), min(gray.shape[0], i[1]+radius)
        x_s, x_e = max(0, i[0]-radius), min(gray.shape[1], i[0]+radius)
        roi = masked_gray[y_s:y_e, x_s:x_e]

        # Wasser-Suche im ROI (oberste 15% ignorieren wegen Becherrand)
        margin = int(roi.shape[0] * 0.15)
        search_area = roi[margin:, :]
        _, binary = cv2.threshold(search_area, wasser_thresh, 255, cv2.THRESH_BINARY_INV)
        
        row_sums = np.sum(binary, axis=1)
        water_pos = np.where(row_sums > (binary.shape[1] * 0.25))[0]

        # Zeichnen
        cv2.circle(img_array, center, radius, (255, 255, 0), 5) # Gelber Kreis
        
        if len(water_pos) > 0:
            top_y = y_s + water_pos[0] + margin
            cv2.line(img_array, (x_s, top_y), (x_e, top_y), (0, 0, 255), 5) # Blaue Linie
            
            prozent = 100 - ((water_pos[0] + margin) / (2 * radius) * 100)
            st.metric("Füllstand", f"{max(0, min(100, prozent)):.1f} %")
        else:
            st.metric("Füllstand", "0.0 %")

        st.image(img_array, caption="Analyse")
    else:
        st.warning("Kein Becher gefunden. Bitte mittig fotografieren.")