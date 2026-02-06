import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Wasserstand Detektor", layout="centered")
st.title("🥤 Wasserstand-Analyse Pro (Ellipsen-Modus)")

# --- Parameter ---
st.sidebar.header("Einstellungen")
wasser_thresh = st.sidebar.slider("Wasser-Schwellenwert (Dunkelheit)", 0, 255, 100, help="Höher = erkennt helleren Kaffee/Tee")
# NEU: Regler für die perspektivische Stauchung
perspective_ratio = st.sidebar.slider("Perspektive (Blickwinkel)", 0.1, 1.0, 0.6, 0.05, help="1.0 = Kreis (Draufsicht), 0.1 = Flache Ellipse (Seitenansicht)")

img_file = st.camera_input("Foto machen")

if img_file is not None:
    image = Image.open(img_file)
    img_array = np.array(image)
    # BGR Konvertierung für OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Robuste Kreis-Erkennung (findet den Becher)
    gray_blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=50, maxRadius=400)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        i = circles[0, 0]
        center_x, center_y = i[0], i[1]
        radius = i[2]

        # --- MASKE ERSTELLEN (Bleibt basierend auf dem Kreis) ---
        # Wir nutzen den Kreis für die Maske, da dies am robustesten ist,
        # um den Innenraum zu isolieren.
        mask = np.zeros_like(gray)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        masked_gray = cv2.bitwise_and(gray, mask)

        # Bereich (ROI) ausschneiden
        y_s, y_e = max(0, center_y - radius), min(gray.shape[0], center_y + radius)
        x_s, x_e = max(0, center_x - radius), min(gray.shape[1], center_x + radius)
        roi = masked_gray[y_s:y_e, x_s:x_e]

        # --- WASSERSTAND ANALYSE ---
        # Oberste 15% des Kreises ignorieren (Becherrand-Schatten)
        margin = int(roi.shape[0] * 0.15)
        search_area = roi[margin:, :]
        
        # Schwellenwert anwenden
        _, binary = cv2.threshold(search_area, wasser_thresh, 255, cv2.THRESH_BINARY_INV)
        
        row_sums = np.sum(binary, axis=1)
        # Eine Zeile gilt als Wasserlinie, wenn sie zu >25% dunkel ist
        water_pos = np.where(row_sums > (binary.shape[1] * 0.25))[0]

        # --- VISUALISIERUNG MIT ELLIPSEN ---
        # Definition der Ellipsen-Achsen basierend auf Radius und Perspektive
        # Hauptachse = Radius, Nebenachse = Radius * Perspektiv-Faktor
        axes_length = (radius, int(radius * perspective_ratio))
        angle = 0 # Drehung der Ellipse (0 Grad bei horizontalem Foto)

        # 1. GELBE ELLIPSE (Becherrand) statt Kreis zeichnen
        # Wir zeichnen eine volle Ellipse (0 bis 360 Grad)
        cv2.ellipse(img_array, (center_x, center_y), axes_length, angle, 0, 360, (255, 255, 0), 4)
        
        if len(water_pos) > 0:
            # Y-Position der Wasserlinie berechnen
            top_water_rel = water_pos[0] + margin
            top_water_abs = y_s + top_water_rel
            
            # 2. BLAUE ELLIPSE (Wasserstand) statt Linie
            # Wir zeichnen nur den vorderen Bogen (0 bis 180 Grad), das sieht echter aus.
            # Das Zentrum der Wasser-Ellipse liegt auf der gleichen X-Achse, aber tiefer auf Y.
            water_center = (center_x, top_water_abs)
            
            # Optional: Die Wasser-Ellipse etwas kleiner machen, da der Becher nach unten enger wird.
            # Hier vereinfacht nutzen wir die gleichen Achsen wie oben.
            cv2.ellipse(img_array, water_center, axes_length, angle, 0, 180, (0, 0, 255), 4)
            
            # Prozentberechnung (linear basierend auf der Höhe im Kreis)
            prozent = 100 - (top_water_rel / (2 * radius) * 100)
            st.metric("Füllstand (ca.)", f"{max(0, min(100, prozent)):.1f} %")
        else:
            st.metric("Füllstand", "Leer / Nicht erkannt")

        st.image(img_array, caption="Analyse mit Ellipsen")
    else:
        st.warning("Kein Becher erkannt. Bitte mittig positionieren.")