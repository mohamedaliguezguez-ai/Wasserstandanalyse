import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- KONFIGURATION ---
st.set_page_config(page_title="Kreis- & Liniendetektor", layout="wide")
st.title("⭕📏 Universal-Detektor: Kreise und Linien")
st.write("Lade ein Bild hoch. Das Programm versucht, alle Kreise (Grün) und Linien (Blau) zu finden.")

# --- SIDEBAR EINSTELLUNGEN ---
st.sidebar.header("⚙️ Einstellungen")

st.sidebar.subheader("1. Vorverarbeitung (Wichtig!)")
# Canny ist entscheidend für Linien und hilft auch bei Kreisen
canny_thresh = st.sidebar.slider(
    "Kanten-Schwelle (Canny)", 
    50, 300, 150, 
    help="Bestimmt, was als 'scharfe Kante' gilt. Höher = weniger Kanten."
)

st.sidebar.subheader("2. Kreise (Hough Circles)")
# Parameter für cv2.HoughCircles
dp = st.sidebar.slider("Auflösung (dp)", 0.5, 2.0, 1.2, 0.1, help="Kleiner = genauere Suche, aber langsamer.")
min_dist = st.sidebar.slider("Min. Abstand zwischen Kreisen", 10, 200, 50)
param2 = st.sidebar.slider(
    "Kreis-Empfindlichkeit (param2)", 
    10, 150, 50, 
    help="DER wichtigste Regler für Kreise. Niedriger = findet mehr (auch falsche) Kreise. Höher = findet nur sehr perfekte Kreise."
)
min_radius = st.sidebar.slider("Min. Radius", 0, 100, 20)
max_radius = st.sidebar.slider("Max. Radius", 50, 500, 200)

st.sidebar.subheader("3. Linien (Hough Lines P)")
# Parameter für cv2.HoughLinesP
line_thresh = st.sidebar.slider(
    "Linien-Schwelle (Votes)", 
    10, 200, 80,
    help="Wie viele 'Punkte' müssen auf einer Linie liegen, damit sie erkannt wird. Niedriger = mehr Linien."
)
min_line_len = st.sidebar.slider("Min. Linienlänge", 10, 200, 50)
max_line_gap = st.sidebar.slider(
    "Max. Lücke in Linie", 
    1, 50, 20,
    help="Erlaubt Unterbrechungen in einer Linie. Größer = verbindet gestrichelte Linien eher."
)


# --- HAUPTPROGRAMM ---
uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Bild laden und vorbereiten
    image_pil = Image.open(uploaded_file)
    img_np = np.array(image_pil)
    # Konvertieren für OpenCV (RGB -> BGR)
    img_output = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # Graustufen für die Analyse
    gray = cv2.cvtColor(img_output, cv2.COLOR_BGR2GRAY)
    
    # Rauschunterdrückung (wichtig, damit nicht jeder Krümel als Kante erkannt wird)
    blurred = cv2.medianBlur(gray, 5)

    # --- KREIS-ERKENNUNG ---
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=dp, 
        minDist=min_dist,
        param1=canny_thresh, # Benutzt den Canny-Wert für die interne Kantenerkennung
        param2=param2, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    circle_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circle_count = len(circles[0, :])
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Kreisumriss zeichnen (GRÜN, Dicke 3)
            cv2.circle(img_output, center, radius, (0, 255, 0), 3)
            # Mittelpunkt zeichnen (ROT, Dicke 5)
            cv2.circle(img_output, center, 3, (0, 0, 255), 5)

    # --- LINIEN-ERKENNUNG ---
    # Für Linien brauchen wir zuerst ein reines Kantenbild (Canny)
    edges = cv2.Canny(blurred, canny_thresh // 2, canny_thresh)
    
    lines = cv2.HoughLinesP(
        edges, 
        1, # rho (Abstandsauflösung in Pixeln)
        np.pi/180, # theta (Winkelauflösung in Bogenmaß)
        threshold=line_thresh, 
        minLineLength=min_line_len, 
        maxLineGap=max_line_gap
    )

    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Linie zeichnen (BLAU, Dicke 3)
            cv2.line(img_output, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # --- ANZEIGE ---
    # Zurückkonvertieren nach RGB für Streamlit
    img_rgb_final = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Gefundene Kreise", circle_count)
        st.metric("Gefundene Linien", line_count)
        st.markdown("""
        **Legende:**
        * 🟢 **Grün:** Erkannte Kreise
        * 🔴 **Rot:** Kreismittelpunkte
        * 🔵 **Blau:** Erkannte Linien
        """)
        with st.expander("Kantenbild sehen (Debug)"):
            st.image(edges, caption="Canny Edges (Basis für Linien)")

    with col2:
        st.image(img_rgb_final, caption="Endergebnis", use_container_width=True)

else:
    st.info("Bitte lade ein Bild hoch, um zu beginnen.")