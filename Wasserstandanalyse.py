# ... (dein Code bis zur Kreis-Erkennung) ...

if circles is not None:
    circles = np.uint16(np.around(circles))
    i = circles[0, 0]
    center = (i[0], i[1])
    radius = i[2]

    # --- NEU: EINE MASKE ERSTELLEN ---
    # Wir erstellen ein schwarzes Bild und zeichnen einen weißen gefüllten Kreis darauf
    mask = np.zeros_like(gray)
    cv2.circle(mask, center, radius, 255, -1) # -1 füllt den Kreis aus
    
    # Jetzt machen wir alles außerhalb des Bechers im Graubild komplett SCHWARZ
    masked_gray = cv2.bitwise_and(gray, mask)

    # ROI (Bereich) ausschneiden für die Analyse
    y_start, y_end = max(0, i[1]-radius), min(gray.shape[0], i[1]+radius)
    x_start, x_end = max(0, i[0]-radius), min(gray.shape[1], i[0]+radius)
    roi = masked_gray[y_start:y_end, x_start:x_end]

    if roi.size > 0:
        # Wir ignorieren die obersten 15% (den Rand), da dieser oft Schatten wirft
        margin = int(roi.shape[0] * 0.15)
        search_area = roi[margin:, :]
        
        # Wasser finden (Dunkle Bereiche)
        _, binary = cv2.threshold(search_area, wasser_thresh, 255, cv2.THRESH_BINARY_INV)
        
        row_sums = np.sum(binary, axis=1)
        # Eine Zeile muss zu mindestens 25% "nass" sein
        water_pos = np.where(row_sums > (binary.shape[1] * 0.25))[0]

        if len(water_pos) > 0:
            top_water_rel = water_pos[0] + margin
            top_water_abs = y_start + top_water_rel
            
            # Blau Linie zeichnen (jetzt innerhalb des Kreises)
            cv2.line(img_array, (x_start, top_water_abs), (x_end, top_water_abs), (0, 0, 255), 4)
            
            # Prozent-Rechnung (0% unten, 100% oben am Kreisrand)
            prozent = 100 - (top_water_rel / (2 * radius) * 100)
            st.metric("Füllstand", f"{max(0, min(100, prozent)):.1f} %")
        else:
            st.metric("Füllstand", "0.0 %")