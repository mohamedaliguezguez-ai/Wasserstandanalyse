import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- APP SETUP ---
st.set_page_config(page_title="Auto-Ellipse Pro", layout="centered")
st.title("🥤 Automatische Füllstand-Analyse")
st.write("Erkennt Becher und Wasserstand automatisch als Ellipsen.")

# --- SIDEBAR EINSTELLUNGEN ---
st.sidebar.header("Erkennung anpassen")
# Hilft, wenn "keine Ellipse gefunden" wird
kanten_empfindlichkeit = st.sidebar.slider("Kanten-Schwellenwert", 30, 150, 80)
wasser_dunkelheit = st.sidebar.slider("Wasser-Dunkelheit", 0, 255, 110)

img_file = st.camera_input("Foto machen")

if img_file is not None:
    # Bild laden & vorbereiten
    img = Image.open(img_file)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. KANTEN-DETEKTION (Vorbehandlung)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, kanten_empfindlichkeit // 2, kanten_empfindlichkeit)
    
    # 2. ELLIPSEN-SUCHE (Rim-Detection)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_ellipse = None
    max_area = 0
    
    for cnt in contours:
        if len(cnt) >= 5: # Mindestens 5 Punkte für eine Ellipse nötig
            area = cv2.contourArea(cnt)
            if area > 10000: # Filtert kleine Störgeräusche raus
                ellipse = cv2.fitEllipse(cnt)
                if area > max_area:
                    max_area = area
                    best_ellipse = ellipse

    # 3. ANALYSE & ZEICHNEN
    if best_ellipse:
        # Gelbe Ellipse für den Rand
        cv2.ellipse(img_array, best_ellipse, (255, 255, 0), 5)
        
        # --- WASSERSTAND FINDEN ---
        # Wir isolieren den Bereich innerhalb der Ellipse
        mask = np.zeros_like(gray)
        cv2.ellipse(mask, best_ellipse, 255, -1)
        masked_gray = cv2.bitwise_and(gray, mask)
        
        # Wir suchen die oberste dunkle Kante im Becher
        # Wir ignorieren die obersten 15% (wegen Schatten am Rand)
        (xc, yc), (d1, d2), angle = best_ellipse
        h_halbe = max(d1, d2) / 2
        y_start = int(yc - h_halbe * 0.7)
        y_end = int(yc + h_halbe)
        
        roi = masked_gray[max(0, y_start):y_end, :]
        if roi.size > 0:
            _, binary = cv2.threshold(roi, wasser_dunkelheit, 255, cv2.THRESH_BINARY_INV)
            row_sums = np.sum(binary, axis=1)
            water_indices = np.where(row_sums > (binary.shape[1] * 0.3))[0]
            
            if len(water_indices) > 0:
                # Wasser-Ellipse zeichnen
                water_y_abs = y_start + water_indices[0]
                # Die Wasser-Ellipse übernimmt Winkel und Form vom Rand
                water_ellipse = ((xc, water_y_abs), (d1 * 0.95, d2 * 0.95), angle)
                cv2.ellipse(img_array, water_ellipse, (0, 0, 255), 4)
                
                # Füllstand berechnen
                total_h = y_end - y_start
                fill_pct = 100 - (water_indices[0] / total_h * 100)
                st.metric("Füllstand", f"{max(0, min(100, fill_pct)):.1f} %")
            else:
                st.info("Kein Wasser gefunden. Prüfe die Dunkelheit-Einstellung.")
        
        st.image(img_array, caption="Analyse mit Auto-Ellipse")
    else:
        st.error("Kein Becher-Rand erkannt. Bitte näher ran oder Kanten-Regler anpassen.")