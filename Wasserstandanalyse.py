import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Wasserstand Pro: Auto-Ellipse", layout="centered")
st.title("🥤 Automatische Ellipsen-Analyse")

# --- SIDEBAR ---
st.sidebar.header("Erkennungs-Parameter")
canny_low = st.sidebar.slider("Kanten-Empfindlichkeit (Low)", 10, 100, 50)
canny_high = st.sidebar.slider("Kanten-Empfindlichkeit (High)", 100, 300, 150)
wasser_thresh = st.sidebar.slider("Wasser-Schwellenwert", 0, 255, 100)

img_file = st.camera_input("Foto der Tasse machen")

if img_file is not None:
    # Bild laden
    img = Image.open(img_file)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Vorverarbeitung für die Kanten-Erkennung
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)
    
    # 2. Konturen finden
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_ellipse = None
    max_area = 0

    # 3. Alle Konturen prüfen und die beste Ellipse finden
    for cnt in contours:
        if len(cnt) >= 5:  # cv2.fitEllipse benötigt mindestens 5 Punkte
            area = cv2.contourArea(cnt)
            if area > 5000:  # Ignoriere zu kleine Fragmente
                ellipse = cv2.fitEllipse(cnt)
                # Wir suchen die größte Ellipse (meist der Becherrand)
                if area > max_area:
                    max_area = area
                    best_ellipse = ellipse

    if best_ellipse:
        # Gelbe Ellipse für den Becherrand zeichnen
        cv2.ellipse(img_array, best_ellipse, (255, 255, 0), 5)
        
        # --- WASSERSTAND ANALYSE ---
        # Zentrum und Achsen der gefundenen Ellipse
        (xc, yc), (d1, d2), angle = best_ellipse
        r_eff = max(d1, d2) / 2
        
        # Maske erstellen, um nur IM Becher zu suchen
        mask = np.zeros_like(gray)
        cv2.ellipse(mask, best_ellipse, 255, -1)
        masked_gray = cv2.bitwise_and(gray, mask)
        
        # Bereich unterhalb des oberen Randes prüfen
        # Wir ignorieren die obersten 20% der Ellipse (Schaum/Schatten)
        y_start = int(yc - r_eff * 0.6)
        y_end = int(yc + r_eff)
        x_start = int(xc - r_eff)
        x_end = int(xc + r_eff)
        
        roi = masked_gray[max(0, y_start):y_end, max(0, x_start):x_end]
        
        if roi.size > 0:
            _, binary = cv2.threshold(roi, wasser_thresh, 255, cv2.THRESH_BINARY_INV)
            row_sums = np.sum(binary, axis=1)
            water_indices = np.where(row_sums > (binary.shape[1] * 0.3))[0]
            
            if len(water_indices) > 0:
                # Wasserlinie gefunden
                rel_y = water_indices[0]
                abs_y = y_start + rel_y
                
                # Blaue Ellipse (Wasserfläche) zeichnen
                # Wir nehmen die gleiche Breite/Winkel wie oben, aber tiefer
                water_ellipse = ((xc, abs_y), (d1 * 0.9, d2 * 0.9), angle)
                cv2.ellipse(img_array, water_ellipse, (0, 0, 255), 4)
                
                # Füllstandsberechnung
                height_total = y_end - y_start
                level = 100 - (rel_y / height_total * 100)
                st.metric("Füllstand automatisch", f"{max(0, min(100, level)):.1f} %")
            else:
                st.info("Keine Wasserfläche im Becher erkannt.")
        
        st.image(img_array, caption="Automatische Ellipsen-Erkennung")
    else:
        st.warning("Konnte keine klare Ellipse finden. Versuche den Becher besser auszuleuchten.")