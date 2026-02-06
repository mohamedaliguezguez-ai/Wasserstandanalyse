import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- APP KONFIGURATION ---
st.set_page_config(page_title="Auto-Ellipsen Analyse", layout="centered")
st.title("🥤 Automatische Füllstand-Analyse")
st.write("Erkennt Becher und Flüssigkeit automatisch als Ellipsen.")

# --- SEITENLEISTE ---
st.sidebar.header("Erkennungs-Feinjustierung")
canny_val = st.sidebar.slider("Kanten-Empfindlichkeit", 10, 200, 80)
wasser_limit = st.sidebar.slider("Dunkelheit der Flüssigkeit", 0, 255, 110)

# --- BILD-UPLOAD / KAMERA ---
img_file = st.camera_input("Foto der Tasse machen")

if img_file is not None:
    # Bild laden
    img = Image.open(img_file)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Bildvorbereitung (Rauschunterdrückung & Kanten)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, canny_val // 2, canny_val)
    
    # 2. Konturen finden
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rim_ellipse = None
    max_area = 0

    # Den größten ovalen Bereich (Becherrand) finden
    for cnt in contours:
        if len(cnt) >= 5: # Mindestens 5 Punkte für fitEllipse nötig
            area = cv2.contourArea(cnt)
            if area > 8000: # Filtert kleine Störungen weg
                ellipse = cv2.fitEllipse(cnt)
                if area > max_area:
                    max_area = area
                    rim_ellipse = ellipse

    if rim_ellipse:
        # Gelbe Ellipse für den Rand zeichnen
        cv2.ellipse(img_array, rim_ellipse, (255, 255, 0), 4)
        
        # --- WASSERSTAND ANALYSE ---
        # Maske erstellen: Alles außerhalb des Bechers ignorieren
        mask = np.zeros_like(gray)
        cv2.ellipse(mask, rim_ellipse, 255, -1)
        masked_gray = cv2.bitwise_and(gray, mask)
        
        # Koordinaten der Rim-Ellipse
        (xc, yc), (d1, d2), angle = rim_ellipse
        half_h = max(d1, d2) / 2
        
        # Nur im Bereich unterhalb des oberen Randes suchen (ca. 80% der Höhe)
        y_start = int(yc - half_h * 0.7)
        y_end = int(yc + half_h)
        roi = masked_gray[max(0, y_start):y_end, :]
        
        if roi.size > 0:
            # Dunkle Bereiche finden (Kaffee/Wasser)
            _, binary = cv2.threshold(roi, wasser_limit, 255, cv2.THRESH_BINARY_INV)
            row_sums = np.sum(binary, axis=1)
            water_indices = np.where(row_sums > (binary.shape[1] * 0.25))[0]
            
            if len(water_indices) > 0:
                # Wasser-Ellipse zeichnen
                water_y = y_start + water_indices[0]
                # Wir nehmen die Form des Randes an, aber tiefer positioniert
                water_ellipse = ((xc, water_y), (d1 * 0.95, d2 * 0.95), angle)
                cv2.ellipse(img_array, water_ellipse, (0, 0, 255), 4)
                
                # Füllstandsberechnung
                fill_height = y_end - y_start
                level = 100 - (water_indices[0] / fill_height * 100)
                st.metric("Füllstand (Auto-Ellipse)", f"{max(0, min(100, level)):.1f} %")
            else:
                st.warning("Flüssigkeit nicht klar erkannt. Schwellenwert anpassen.")
        
        st.image(img_array, caption="Analyse-Ergebnis")
    else:
        st.error("Kein Becher-Rand erkannt. Bitte näher ran oder Kanten-Regler anpassen.")