# Importar paquetes
# https://www.youtube.com/watch?v=p3yfEtFQEEw
import cv2
from tqdm import *
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(path_video_in, path_video_out):
    """
    Procesa un video para calcular métricas, dibujar trayectorias y generar un video de salida.
    :param path_video_in: Ruta del video de entrada (str).
    :param path_video_out: Ruta de archivos de salida (str).
    """
    global metrics  # Definición global de métricas

    video_in = cv2.VideoCapture(path_video_in) # Video entrada

    # Propiedades del video para video salida
    fps = video_in.get(cv2.CAP_PROP_FPS)
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Video de salida
    video_out = cv2.VideoWriter(path_video_out + '/video_out.mp4', fourcc, fps, (width, height))

    # Se inicia barra progreso
    progress_bar = tqdm(total=total_frames, desc="Processing Frames")
    last_frame = None  # último frame

    while True:
        ret, frame = video_in.read()
        track_layer = np.zeros_like(frame) # Capa para dibujar la trayectoria
        if not ret:
            break

        img_filt = frame_processing(frame, track_layer, fps) # Se procesa el frame
        last_frame = img_filt  # Guardar el último frame

        video_out.write(img_filt)  # Escribir video
        progress_bar.update(1) # Actualizar barra

    metrics['Velocidad angular media (vueltas/s)'] = round(sum(total_vel_w) / len(total_vel_w),2) #Actualizar métricas

    if last_frame is not None:  # último frame
        table_frame = metrics_pandas(last_frame)  # Dibujar métricas en video
        for _ in range(int(fps * 5)):  # Congelar por 5 segundos
            video_out.write(table_frame)  # Escribir video salida

    # Liberar recursos
    video_in.release()
    video_out.release()
    cv2.destroyAllWindows()
    draw_graph(path_out)  # Dibujar gráfica
    progress_bar.close()

def metrics_pandas(frame):
    """
    Convierte métricas en una tabla y las añade al frame.

    :param frame: Imagen sobre la cual se dibujará la tabla (ndarray).
    :return frame: Frame con tabla (ndarray).
    """
    global metrics # Métricas
    # Convertir a dataframe y a texto
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metrica", "Valor"])
    metrics_text = metrics_df.to_string(index=False)

    # Coordenadas tabla
    x, y = 50, 100
    line_height = 30

    # Dibujar un rectángulo semitransparente como fondo de la tabla
    back_g = frame.copy()
    cv2.rectangle(back_g, (x - 20, y - 30), (x + 460, 120 + len(metrics_df) * line_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(back_g, 0.5, frame, 0.5, 0)

    # Propiedades texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # Blanco
    thickness = 1

    # Añadir líneas de la tabla al frame
    for i, line in enumerate(metrics_text.split("\n")):
        cv2.putText(frame, line, (x, y + i * line_height), font, font_scale, color, thickness)

    return frame

def frame_processing(frame, track_layer, fps):
    """
    Dado un frame, una capa para el dibujo de la trayectoria y los fps,
    se procesa el video, se dibuja la trayectoria y se obtienen las métricas.
    :param frame: Frame del video (ndarray).
    :param track_layer: Capa trayectoria (ndarray).
    :param fps: Fps del video (int).
    :return results: Resultado del frame procesado (ndarray).
    """
    global low_h, high_h

    img_arr = Image.fromarray(frame)
    # Aplicar un filtro de nitidez
    img_filt = img_arr.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    img_filt_arr = np.array(img_filt)

    low_h_np = np.array(low_h, dtype=np.uint8)
    high_h_np = np.array(high_h, dtype=np.uint8)

    # Máscara con umbrales hsv
    hsv_img = cv2.cvtColor(img_filt_arr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, low_h_np, high_h_np)
    # Erosión y dilatación
    eros_img = erosion_img(mask, (3, 3), 1)
    dilat_img = dilation_img(eros_img, (15, 15), 5)

    pxl_cont = contour_processing(frame, dilat_img) # Procesar contorno
    # Procesar trayecoria
    results = trayectoria_processing(pxl_cont, frame, track_layer, fps)

    return results

def trayectoria_processing(list_pxl, original_img, track_layer, fps):
    """
    Calcula y dibuja la trayectoria dado un contorno.
    :param list_pxl: Lista de contorno detectado (list).
    :param original_img: Imagen original (ndarray).
    :param track_layer: Capa trayectoria (ndarray).
    :param fps: Fps del video (int).
    :return mix_img: Imagen original y dibujada (ndarray).
    """
    global track, time, graph, scale
    m = cv2.moments(list_pxl) # calcular momentos del contorno
    if m['m00'] != 0:
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        centroid = (cx, cy)  # Calcular centroide
    else:
        # Si no hay momentos válidos, calcular el círculo mínimo envolvente
        (x_axis, y_axis), radius = cv2.minEnclosingCircle(list_pxl)
        center = (int(x_axis), int(y_axis))
        centroid = center

    if track:
        # Calcular métricas
        calculate_metrics(centroid, original_img, track, fps)

    track.append(centroid) # Actualizar la lista de trayectoria
    if len(track) > 15:  # Solo hacer el seguimiento en 15 puntos
        track.pop(0)

    for i in range(1, len(track)):
        # Dibujar
        cv2.line(track_layer, track[i - 1], track[i], (0, 255, 127), thickness=8)

    # Sumar capa trayectoria y frmae
    mix_img = cv2.addWeighted(original_img, 1, track_layer, 0.85, 10)
    # tiempo y posición para gráfica
    graph.append([time, (original_img.shape[0] - centroid[1]) * scale])
    time += 1/fps
    return mix_img

def contour_processing(original_img, img_pre):
    """
    Encuentra el contorno más grande.
    :param original_img: Imagen original (ndarray).
    :param img_pre: Imagen preprocesada (ndarray).
    :return engl_cont: Contorno convexo más grande (ndarray).
    """
    contours, _ = cv2.findContours(img_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    engl_cont = cv2.convexHull(max_contour) # Envolver el contorno
    # Para dibujar
    #cv2.drawContours(original_img, [engl_cont], -1, 255, thickness=cv2.FILLED)

    return engl_cont

def calculate_metrics(centroid_px, original_img, track_curr, fps):
    """
    Calcula metricas de la trayectoria seguida.
    :param centroid_px: Centroide del contorno (tuple).
    :param original_img: Imagen original (ndarray).
    :param track_curr: Lista de puntos de la trayectoria (list).
    :param fps: Fps del video (int).
    """
    global scale, total_vel_w, alpha_prev, metrics

    # Métricas
    metrics['Altura maxima (m)'] = max(metrics['Altura maxima (m)'], round((original_img.shape[0] - centroid_px[1]) * scale , 2))

    # Cambios en la posición angular
    prev_point = track_curr[-1]
    dy = centroid_px[1] - prev_point[1]
    dx = centroid_px[0] - prev_point[0]
    alpha = np.arctan2(dy, dx) # Ángulo en radianes

    if alpha_prev is not None:
        # Calcular delta angular
        delta_alpha = alpha - alpha_prev

        # Asegurar que delta_alpha esté dentro de [-pi, pi]
        delta_alpha = np.arctan2(np.sin(delta_alpha), np.cos(delta_alpha))

        # Calcular tiempo entre frames
        delta_time = 1 / fps

        # Calcular velocidad angular (vueltas por segundo)
        vel_w = abs(delta_alpha) / (2 * np.pi * delta_time)

        # Actualizar métricas
        metrics['Velocidad angular maxima (vueltas/s)'] = max(metrics['Velocidad angular maxima (vueltas/s)'], round(vel_w, 2))
        total_vel_w.append(round(vel_w,2))

        # Actualizar el ángulo previo
    alpha_prev = alpha

def draw_graph(path_out):
    """
    Genera gráfica de posición respecto al tiempo
    :param path_out: Ruta del archivo (str).
    """
    global graph
    # Datos tiempo y posición metros
    x = [i[0] for i in graph]
    y = [i[1] for i in graph]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linestyle='-', linewidth=2.5, label='Posición vs Tiempo', color='r')
    plt.title('Posición en función del Tiempo', fontsize=16, fontweight='bold')
    plt.xlabel('Tiempo (s)', fontsize=14)
    plt.ylabel('Posición (m)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Guardar la gráfica
    plt.savefig(path_out + '/gaph_mov.png', dpi=300)  # Guardar con alta resolución
    plt.close()  # Cerrar la figura

def erosion_img(img_input, kernel, iterations):
    """
    Aplica erosión
    :param img_input: Imagen entrada (ndarray).
    :param kernel: Tamaño del kernel (tuple).
    :param iterations: Iteraciones (int).
    :return erosion: Imagen procesada (ndarray).
    """
    kernel = np.ones(kernel, np.uint8)
    erosion = cv2.erode(img_input, kernel, iterations=iterations)
    return erosion

def dilation_img(img_input, kernel, iterations):
    """
    Aplica dilatación
    :param img_input: Imagen entrada (ndarray).
    :param kernel: Tamaño del kernel (tuple).
    :param iterations: Iteraciones (int).
    :return dilation: Imagen procesada (ndarray).
    """
    kernel = np.ones(kernel, np.uint8)
    dilation = cv2.dilate(img_input, kernel, iterations=iterations)
    return dilation

if __name__ == "__main__":
    path_video = './input/barras_gym.mp4' # Ruta video
    path_out = './output' # Ruta video salida

    low_h = [128, 40, 44]   # Lista valores HSV
    high_h = [160, 139, 165]
    scale = 0.0043  # Escala mppx

    # Métricas y variables para el análisis
    metrics = {'Altura maxima (m)': float(0), 'Velocidad angular media (vueltas/s)': float(0), 'Velocidad angular maxima (vueltas/s)':float(0)}
    time = 0
    track = []
    graph = []
    total_vel_w = []
    alpha_prev = None

    main(path_video, path_out )
