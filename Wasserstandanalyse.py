import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Wasserstand Pro", layout="wide")
st.title("🥤 Intelligente Becher-Analyse")

# --- SIDEBAR FÜR DIE FEINEINSTELLUNG ---
st.sidebar.header("Analyse-Tuning")
kanten_starke = st.sidebar.slider("Kanten-Filter (Becher finden)", 10, 250, 80)
wasser_limit = st.sidebar.slider("Flüssigkeits-Kontrast", 0, 255, 110)

img_file = st.camera_input("Foto der Tasse")

if img_file:
    img = Image.open(img_file)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Kanten finden & Silhouette vorbereiten
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, kanten_starke // 2, kanten_starke)
    
    # 2. Konturen suchen
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    silhouette_view = np.zeros_like(gray) # Ansicht für die Silhouette
    found_cup = False

    if contours:
        # Die größte geschlossene Form suchen
        cup_contour = max(contours, key=cv2.contourArea)
        
        # Nur weitermachen, wenn die Form groß genug ist (kein Rauschen)
        if cv2.contourArea(cup_contour) > 5000:
            found_cup = True
            cv2.drawContours(silhouette_view, [cup_contour], -1, 255, -1)
            
            if len(cup_contour) >= 5:
                # Becher-Ellipse berechnen
                ellipse = cv2.fitEllipse(cup_contour)
                (xc, yc), (d1, d2), angle = ellipse
                cv2.ellipse(img_array, ellipse, (255, 255, 0), 5)

                # --- WASSER-SUCHE (Nur innerhalb der Silhouette) ---
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cup_contour], -1, 255, -1)
                roi = cv2.bitwise_and(gray, mask)

                # Suchbereich (oberste 20% ignorieren)
                y_start = int(yc - (max(d1, d2) / 2) * 0.7)
                y_end = int(yc + (max(d1, d2) / 2))
                search_area = roi[max(0, y_start):y_end, :]
                
                if search_area.size > 0:
                    _, binary = cv2.threshold(search_area, wasser_limit, 255, cv2.THRESH_BINARY_INV)
                    row_sums = np.sum(binary, axis=1)
                    water_line = np.where(row_sums > (binary.shape[1] * 0.3))[0]

                    if len(water_line) > 0:
                        y_abs = y_start + water_line[0]
                        # Wasser-Ellipse zeichnen
                        water_ell = ((xc, y_abs), (d1 * 0.9, d2 * 0.4), angle)
                        cv2.ellipse(img_array, water_ell, (0, 0, 255), 4)
                        
                        level = 100 - (water_line[0] / (y_end - y_start) * 100)
                        st.metric("Füllstand", f"{max(0, min(100, level)):.1f} %")

    # --- ANZEIGE DER ERGEBNISSE ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Analyse")
        st.image(img_array)
    with col2:
        st.subheader("Was der Computer sieht (Silhouette)")
        st.image(silhouette_view)
        if not found_cup:
            st.error("Keine Silhouette erkannt! Tipp: Stell den Becher auf einen dunklen Untergrund.")

else:
    st.info("Bitte mache ein Foto. Achte auf guten Kontrast (z.B. dunkler Tisch).")