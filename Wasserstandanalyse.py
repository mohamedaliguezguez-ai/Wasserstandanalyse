import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Silhouette Mess-App", layout="wide")
st.title("🔬 Becher-Analyse mit Silhouetten-Ansicht")

# --- SIDEBAR ---
st.sidebar.header("Analyse-Tuning")
kanten_empf = st.sidebar.slider("Kanten-Empfindlichkeit", 10, 255, 100, help="Niedriger = mehr Details, Höher = nur starke Kanten")
wasser_limit = st.sidebar.slider("Flüssigkeits-Kontrast", 0, 255, 110)

img_file = st.camera_input("Foto der Tasse")

if img_file:
    # Bildvorbereitung
    img = Image.open(img_file)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Kanten-Bild erstellen (Canny)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, kanten_empf // 2, kanten_empf)
    
    # 2. Konturen finden
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Platzhalter für die Silhouette
    silhouette_view = np.zeros_like(gray)
    
    if contours:
        # Die größte Form finden (der Becher)
        cup_contour = max(contours, key=cv2.contourArea)
        
        # Silhouette zeichnen (Weiß auf Schwarz)
        cv2.drawContours(silhouette_view, [cup_contour], -1, 255, -1)
        
        # Ellipse an die Silhouette anpassen
        if len(cup_contour) >= 5:
            ellipse = cv2.fitEllipse(cup_contour)
            (xc, yc), (d1, d2), angle = ellipse
            cv2.ellipse(img_array, ellipse, (255, 255, 0), 5) # Gelber Rand

            # Wasser-Suche (innerhalb der Silhouette)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cup_contour], -1, 255, -1)
            roi = cv2.bitwise_and(gray, mask)
            
            # Analyse-Bereich (oberen Rand ignorieren)
            y_start = int(yc - (max(d1, d2) / 2) * 0.7)
            y_end = int(yc + (max(d1, d2) / 2))
            search_area = roi[max(0, y_start):y_end, :]
            
            if search_area.size > 0:
                _, binary = cv2.threshold(search_area, wasser_limit, 255, cv2.THRESH_BINARY_INV)
                row_sums = np.sum(binary, axis=1)
                water_line = np.where(row_sums > (binary.shape[1] * 0.3))[0]

                if len(water_line) > 0:
                    y_abs = y_start + water_line[0]
                    # Wasser-Ellipse zeichnen (Blau)
                    water_ell = ((xc, y_abs), (d1 * 0.9, d2 * 0.4), angle)
                    cv2.ellipse(img_array, water_ell, (0, 0, 255), 4)
                    
                    level = 100 - (water_line[0] / (y_end - y_start) * 100)
                    st.metric("Füllstand", f"{max(0, min(100, level)):.1f} %")

    # --- ANZEIGE DER ERGEBNISSE ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Analyse")
        st.image(img_array, use_container_width=True)
        
    with col2:
        st.subheader("2. Kanten (Canny)")
        st.image(edged, use_container_width=True)
        st.caption("Alle Linien, die das Handy erkennt.")
        
    with col3:
        st.subheader("3. Silhouette")
        st.image(silhouette_view, use_container_width=True)
        st.caption("Das ist die Form, die als Becher zählt.")

else:
    st.info("Bitte mache ein Foto, um die Silhouette zu sehen.")