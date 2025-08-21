# Importar paquetes
import cv2


def interfaz_escala(image_in):
    """
    Muestra una interfaz gráfica para seleccionar dos puntos en la imagen y calcular su escala.
    :param image_in: Ruta de la imagen (str).
    :return points: Coordendas (x,y) de los puntos seleccionados (list).
    """
    window_name = 'Escala Px'

    def click_event(event, x, y, flags, param):
        """
        Maneja la ventana de la interfaz y las acciones según los click del ratón.
        :param event: Click event, se usará click izquierdo.
        :param x: Coordenada X del click
        :param y: Coordenada Y del click
        :param flags:
        :param param:
        :return:
        """
        nonlocal points # lista de puntos

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

            cv2.circle(cop_image, (x, y), 5, (0, 0, 255), -1) # Se dibuja círculo rojo
            cv2.imshow(window_name, cop_image)
            if len(points) == 2:
                # Si hay dos puntos se traza una línea
                cv2.line(cop_image, points[0], points[1], (0, 255, 0), 5)
                cv2.imshow(window_name, cop_image)

    # Carga la imagen
    image = cv2.imread(image_in)

    # Redimensionar la ventana para que sea ajustable
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    height, width = image.shape[:2]

    scale_w = 0.7 # Escala la ventana
    new_width = int(width * scale_w)
    new_height = int(height * scale_w)
    cv2.resizeWindow(window_name, new_width, new_height) # Redimensiona

    # Copia de la imagen para mostrar los puntos seleccionados
    cop_image = image.copy()
    points = []

    # Mostrar la imagen
    cv2.imshow(window_name, cop_image)
    cv2.setMouseCallback(window_name, click_event)
    cv2.waitKey(0)  # Esperar a presionar tecla
    cv2.destroyAllWindows()

    if len(points) != 2: # Solo selección de dos puntos
        return None
    return points


def calculate(px_points, h_m):
    """
    Calcula la escala en metros por píxel usando los puntos de referencia y una altura conocida.

    :param px_points: Lista de dos tuplas, cada una representando un punto (x, y) en píxeles (list).
    :param h_m: Altura conocida en metros (float).
    :return: None
    """
    (x1, y1), (x2, y2) = px_points # Coordendas puntos
    h_bar_px = abs(y2 - y1) # Diferencia en valor abs
    scale_px = h_m / h_bar_px # Escala metros por pixel
    print(f'Altura de la barra: {h_bar_px} píxeles')
    print(f'Escala: {scale_px:.4f} metros por píxel')


if __name__ == "__main__":
    image_path = './HSV_Scale/try_hsv.png'  # Path imagen de entrada
    reference_points = interfaz_escala(image_path) #Puntos de inicio y fin en base a referencia
    h_bar_m = 2.4  # Distancia conocida
    if reference_points:
        calculate(reference_points, h_bar_m)  # Calcula la escala
