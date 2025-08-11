import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import (
    threshold_otsu,
    threshold_li,
    threshold_niblack,
    threshold_sauvola
)

def plot_thresholds(img1_valid, output_path_th, window_size=31, k=0.2, r=None, extent=None):
    """
    Calcula y visualiza las máscaras binarias (Otsu, Li, Niblack, Sauvola) con colores personalizados
    y ejes en coordenadas geográficas si se proporciona `extent`.
    
    Args:
        img1_valid (np.ndarray): Imagen 2D en escala de grises (float).
        output_path_th (str): Ruta de salida para guardar la imagen.
        window_size (int): Tamaño de vecindad para Niblack/Sauvola.
        k (float): Parámetro de sensibilidad.
        r (float | None): Rango dinámico en Sauvola.
        extent (list | None): [xmin, xmax, ymin, ymax] para usar ejes geográficos.
    """
    # --- OTSU ---
    t_otsu = threshold_otsu(img1_valid)
    mask_otsu = (img1_valid > t_otsu).astype(np.float32)

    # --- LI ---
    t_li = threshold_li(img1_valid)
    mask_li = (img1_valid > t_li).astype(np.float32)

    # --- NIBLACK ---
    t_niblack_map = threshold_niblack(img1_valid, window_size=window_size, k=k)
    t_niblack_global = np.mean(t_niblack_map)
    mask_niblack = (img1_valid > t_niblack_global).astype(np.float32)

    # --- SAUVOLA ---
    t_sauvola_map = threshold_sauvola(img1_valid, window_size=window_size, k=k, r=r)
    t_sauvola_global = np.mean(t_sauvola_map)
    mask_sauvola = (img1_valid > t_sauvola_global).astype(np.float32)

    print(f"Umbral Otsu: {t_otsu:.3f}")
    print(f"Umbral Li: {t_li:.3f}")
    print(f"Umbral Niblack (promedio): {t_niblack_global:.3f}")
    print(f"Umbral Sauvola (promedio): {t_sauvola_global:.3f}")

    # Diccionario de máscaras y títulos
    masks = {
        "Otsu": mask_otsu,
        "Li": mask_li,
        "Niblack": mask_niblack,
        "Sauvola": mask_sauvola
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (title, mask) in zip(axes, masks.items()):
        im = ax.imshow(mask, cmap='cool', vmin=0, vmax=1, extent=extent)
        ax.set_title(f"Umbral: {title}", fontsize=12)
        if extent:
            ax.set_xlabel("Longitud")
            ax.set_ylabel("Latitud")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Comparación de Métodos de Umbralización", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path_th, dpi=300)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
