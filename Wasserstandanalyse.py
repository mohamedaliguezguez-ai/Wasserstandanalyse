import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🥤 3D-Zylinder Füllstand-Analyse")
st.write("Erkennt den Becher durch Analyse von Rand, Boden und Seitenwänden.")

# --- SIDEBAR ---
st.sidebar.header("Analyse-Tuning")
sens = st.sidebar.slider("Kanten-Stärke", 20, 200, 100)
wasser_limit = st.sidebar.slider("Flüssigkeit-Dunkelheit", 0, 255, 120)

img_file = st.camera_input("Foto machen")

if img_file is not None:
    img = Image.open(img_file)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Kanten finden (Canny)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, sens // 2, sens)

    # 2. Die größte Kontur finden (Der gesamte Becher)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Die größte Fläche nehmen
        cup_cnt = max(contours, key=cv2.contourArea)
        
        # 3. Eckpunkte bestimmen (Oben, Unten, Seiten)
        # Wir nutzen ein umschließendes Rechteck, um die Extrempunkte zu finden
        x, y, w, h = cv2.boundingRect(cup_cnt)
        
        # Wir definieren die Geometrie
        top_y = y
        bottom_y = y + h
        left_x = x
        right_x = x + w
        center_x = x + w // 2

        # 4. Ellipsen konstruieren
        # Obere Ellipse (Rand)
        rim_ellipse = ((center_x, top_y + 20), (w, 40), 0) # Schätzung der Neigung
        # Untere Ellipse (Boden)
        base_ellipse = ((center_x, bottom_y - 20), (w * 0.8, 30), 0)

        # Zeichnen der Struktur (Gelb)
        cv2.ellipse(img_array, rim_ellipse, (255, 255, 0), 3) # Oben
        cv2.ellipse(img_array, base_ellipse, (255, 255, 0), 2) # Unten
        cv2.line(img_array, (left_x, top_y + 20), (left_x + 20, bottom_y - 20), (255, 255, 0), 2) # Seite Links
        cv2.line(img_array, (right_x, top_y + 20), (right_x - 20, bottom_y - 20), (255, 255, 0), 2) # Seite Rechts

        # 5. WASSERSTAND IN DIESEM BEREICH SUCHEN
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cup_cnt], -1, 255, -1)
        roi_gray = cv2.bitwise_and(gray, mask)
        
        # Nur im mittleren Bereich des Bechers suchen
        search_roi = roi_gray[top_y + 40 : bottom_y - 10, left_x : right_x]
        
        if search_roi.size > 0:
            _, binary = cv2.threshold(search_roi, wasser_limit, 255, cv2.THRESH_BINARY_INV)
            row_sums = np.sum(binary, axis=1)
            water_idx = np.where(row_sums > (binary.shape[1] * 0.3))[0]

            if len(water_idx) > 0:
                water_y_abs = top_y + 40 + water_idx[0]
                # Wasser-Ellipse zeichnen (Blau)
                # Breite wird proportional zum Füllstand schmaler (falls konisch)
                width_factor = 1.0 - (water_idx[0] / h * 0.2) 
                water_ell = ((center_x, water_y_abs), (int(w * width_factor), 35), 0)
                cv2.ellipse(img_array, water_ell, (0, 0, 255), 4)

                # Berechnung des Füllstands h
                # $h_{rel} = 1 - \frac{y_{water} - y_{top}}{y_{bottom} - y_{top}}$
                level = 100 - (water_idx[0] / (bottom_y - top_y - 50) * 100)
                st.metric("Füllstand", f"{max(0, min(100, level)):.1f} %")

        st.image(img_array, caption="Geometrische 3-Punkt Analyse")
    else:
        st.error("Konnte keine Becher-Silhouette finden. Bitte Hintergrund prüfen.")