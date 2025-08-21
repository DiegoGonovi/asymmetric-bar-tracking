# Importar paquetes
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import numpy as np

class HSVEditor(tk.Frame):
    def __init__(self, master: tk.Tk, image_in: str):  # Especificar el tipo de master
        super().__init__(master)
        self.root = master
        self.root.title("Editor HSV")  # Configurar el título
        self.root.geometry("950x700")

        # Crear el marco superior para la imagen
        self.topframe = tk.Frame(self.root, bg="white", height=500, width=950)
        self.topframe.pack_propagate(False)
        self.topframe.pack(side=tk.TOP, fill="x")

        # Crear el marco inferior para las barras deslizantes
        self.botframe = tk.Frame(self.root, bg="white", height=200)
        self.botframe.pack(side=tk.BOTTOM, fill="both", expand=True)

        # Cargar y ajustar la imagen al tamaño del marco
        self.img_cv = cv2.imread(image_in)
        self.img = Image.open(image_in)
        self.img = self.img.resize((950, 500), Image.Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(self.img)

        self.image_label = tk.Label(self.topframe, image=self.img_tk)
        self.image_label.pack(expand=True, fill="both")

        # Configurar las barras deslizantes usando grid
        self.h_low = tk.Scale(self.botframe, from_=0, to=179, orient=tk.HORIZONTAL, label="H Min",
                                command=self.update_image)
        self.h_low.grid(row=0, column=0, padx=5, pady=2, sticky="ew") # Barra H min

        self.h_high = tk.Scale(self.botframe, from_=0, to=179, orient=tk.HORIZONTAL, label="H Max",
                                 command=self.update_image)
        self.h_high.grid(row=0, column=1, padx=5, pady=2, sticky="ew") # Barra H max

        self.s_low = tk.Scale(self.botframe, from_=0, to=255, orient=tk.HORIZONTAL, label="S Min",
                                command=self.update_image)
        self.s_low.grid(row=1, column=0, padx=5, pady=2, sticky="ew") # Barra S min

        self.s_high = tk.Scale(self.botframe, from_=0, to=255, orient=tk.HORIZONTAL, label="S Max",
                                 command=self.update_image)
        self.s_high.grid(row=1, column=1, padx=5, pady=2, sticky="ew") # Barra S max

        self.v_low = tk.Scale(self.botframe, from_=0, to=255, orient=tk.HORIZONTAL, label="V Min",
                                command=self.update_image)
        self.v_low.grid(row=2, column=0, padx=5, pady=2, sticky="ew") # Barra V min

        self.v_high = tk.Scale(self.botframe, from_=0, to=255, orient=tk.HORIZONTAL, label="V Max",
                                 command=self.update_image)
        self.v_high.grid(row=2, column=1, padx=5, pady=2, sticky="ew") # Barra V max

        # Configurar tamaños dinámicos para las columnas
        self.botframe.grid_columnconfigure(0, weight=1)
        self.botframe.grid_columnconfigure(1, weight=1)

    def update_image(self, event=None):
        """
        Dado el valor de las barras dinámicas HSV mostrar la imagen en el espacio de color HSV definido.
        """
        # Convierte la imagen a HSV
        hsv_image = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2HSV)

        lower_val = np.array([self.h_low.get(), self.s_low.get(), self.v_low.get() ])
        upper_val = np.array([self.h_high.get(), self.s_high.get(), self.v_high.get() ])

        # Aplica la máscara con los valores proporcionados
        mask = cv2.inRange(hsv_image, lower_val, upper_val)
        mask_pil = Image.fromarray(mask).resize((950, 500), Image.Resampling.LANCZOS) # Formato pil para mostrar
        self.img_tk = ImageTk.PhotoImage(mask_pil)
        self.image_label.config(image=self.img_tk)
        self.image_label.image = self.img_tk


if __name__ == "__main__":
    # Ruta de la imagen
    image_input = './HSV_Scale/try_hsv.png'  # Imagen para evaluar el hsv

    # Crear la ventana principal
    root = tk.Tk()
    app = HSVEditor(root, image_input)
    root.mainloop()
