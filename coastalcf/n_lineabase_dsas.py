# coastalcf/plots.py

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
from esa_snappy import jpy



def plot_binary_map(binary_data, output_path_mask, metodo=""):
    """
    Dibuja y guarda la máscara binaria de detección de agua.
    """
    plt.figure(figsize=(7, 7))
    titulo = f'Mapa binario de detección de agua ({metodo})' if metodo else 'Mapa binario de detección de agua'
    plt.title(titulo)
    plt.imshow(binary_data, cmap='gray', vmin=0, vmax=1)
    plt.savefig(output_path_mask, dpi=300)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def plot_histogram(band_data, output_path_hist, metodo=""):
    """
    Dibuja y guarda el histograma de la banda.
    """
    titulo = f'Histograma de la banda binaria ({metodo})' if metodo else 'Histograma de la banda binaria'
    plt.hist(band_data, bins=512, range=[0, 15], density=True)
    plt.title(titulo)
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia normalizada')
    plt.savefig(output_path_hist, dpi=300)
    plt.show(block=False)
    plt.pause(2)
    plt.close()




PixelPos = jpy.get_type('org.esa.snap.core.datamodel.PixelPos')
GeoPos = jpy.get_type('org.esa.snap.core.datamodel.GeoPos')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FormatStrFormatter
import os
from esa_snappy import PixelPos, GeoPos

"""
def aplicar_correccion_y_grid(productos_binarios, output_directory, sentinel_id=""):
  
    #Muestra en un grid los productos corregidos geométricamente con ejes lat/lon y colores personalizados.
  
    # Colores: -1 = blanco (nodata), 0 = deeppink (agua), 1 = goldenrod (tierra)
    colors = ['white', 'deeppink', 'goldenrod']
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    num = len(productos_binarios)
    ncols = 2
    nrows = int(np.ceil(num / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
    axes = axes.flatten()

    for i, (nombre, (producto, threshold_val)) in enumerate(productos_binarios.items()):
        band_name_match = [b for b in producto.getBandNames() if nombre in b]
        if not band_name_match:
            print(f"⚠️ No se encontró una banda correspondiente a: {nombre}")
            continue

        band = producto.getBand(band_name_match[0])
        w, h = band.getRasterWidth(), band.getRasterHeight()
        data = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, data)
        data = data.reshape((h, w))

        extent = None
        try:
            geo = producto.getSceneGeoCoding()
            upper_left = GeoPos()
            lower_right = GeoPos()

            geo.getGeoPos(PixelPos(0, 0), upper_left)
            geo.getGeoPos(PixelPos(w - 1, h - 1), lower_right)

            lon_0, lat_0 = upper_left.lon, upper_left.lat
            lon_w, lat_h = lower_right.lon, lower_right.lat
            extent = [lon_0, lon_w, lat_h, lat_0]
        except Exception as e:
            print(f"⚠️ No se pudo obtener geo-referencias para {nombre}: {e}")

        axes[i].imshow(data, cmap=cmap, norm=norm,
                       extent=extent if extent else None, aspect='auto')

        metodo = nombre.replace("flood_", "").capitalize()
        axes[i].set_title(f'{metodo} ({sentinel_id})')
        axes[i].set_xlabel("Longitud")
        axes[i].set_ylabel("Latitud")

        # Rotar y formatear los ticks
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes[i].set_xlim(left=-106.49, right=-106.41)
        axes[i].set_ylim(top=23.265)
        
        axes[i].text(
            0.02, 0.02,  # posición relativa al eje (x, y)
            f"Threshold = {threshold_val:.4f}",
            transform=axes[i].transAxes,
            fontsize=10,
            color='black',
            verticalalignment='bottom',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7)
        )


    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()


    plt.savefig(os.path.join(output_directory, f"{sentinel_id}_ThresholdsCompare.png"), dpi=300)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
"""

def aplicar_correccion_y_grid(productos_binarios, output_directory, sentinel_id="", img_db_product=None, vh_extent=None):
    """
    Muestra en un grid los productos corregidos geométricamente con ejes lat/lon y colores personalizados.
    También grafica la imagen original VH (img_db_product) si se proporciona.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.ticker import FormatStrFormatter
    import os
    from esa_snappy import PixelPos, GeoPos
    titles=['Otsu', 'Niblack', 'Sauvola', 'MCET', 'MCET Iterative']
    def plot_img_db_valid(img_db_product, sentinel_id="", extent=None, ax=None):
        """
        Grafica un producto SNAP (img_db_valid) en dB en un axis dado, usando escala de 0 a 25 dB.
        """
        # Obtener banda
        band = img_db_product.getBand('img_db_valid_VH')
        w, h = band.getRasterWidth(), band.getRasterHeight()

        # Leer pixeles
        img_db_data = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, img_db_data)
        img_db_data = img_db_data.reshape((h, w))

        # --- Plot ---
        im = ax.imshow(img_db_data, cmap='gray', vmin=0, vmax=25,
                    extent=extent if extent else None, aspect='auto')

        ax.set_title(f'Imagen VH en dB {sentinel_id}', fontsize=12)
        ax.set_xlabel('Longitud', fontsize=10)
        ax.set_ylabel('Latitud', fontsize=10)

        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if extent:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        #cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #cbar.set_label('Valor (dB)', fontsize=10)


    # --- Colores de máscara: -1 = blanco (nodata), 0 = cyan (agua), 1 = darkorange (tierra)
    colors = ['white', 'purple', 'pink']
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    num = len(productos_binarios)
    extra_panel = 1 if img_db_product is not None else 0
    total_plots = num + extra_panel

    ncols = 2
    nrows = int(np.ceil(total_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
    axes = axes.flatten()

    # --- Primero graficamos img_db_product en axes[0]
    idx = 0
    if img_db_product is not None:
        plot_img_db_valid(img_db_product, sentinel_id, vh_extent, ax=axes[idx])
        idx += 1

    # --- Luego ploteamos los productos binarios
    for i, (nombre, (producto, threshold_val)) in enumerate(productos_binarios.items()):
        band_name_match = [b for b in producto.getBandNames() if nombre in b]
        if not band_name_match:
            print(f"⚠️ No se encontró una banda correspondiente a: {nombre}")
            continue

        band = producto.getBand(band_name_match[0])
        w, h = band.getRasterWidth(), band.getRasterHeight()
        data = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, data)
        data = data.reshape((h, w))

        extent = None
        try:
            geo = producto.getSceneGeoCoding()
            upper_left = GeoPos()
            lower_right = GeoPos()

            geo.getGeoPos(PixelPos(0, 0), upper_left)
            geo.getGeoPos(PixelPos(w - 1, h - 1), lower_right)

            lon_0, lat_0 = upper_left.lon, upper_left.lat
            lon_w, lat_h = lower_right.lon, lower_right.lat
            extent = [lon_0, lon_w, lat_h, lat_0]
            print(extent)
        except Exception as e:
            print(f"⚠️ No se pudo obtener geo-referencias para {nombre}: {e}")

        axes[idx].imshow(data, cmap=cmap, norm=norm,
                         extent=extent if extent else None, aspect='auto')

        metodo = nombre.replace("flood_", "").capitalize()
        title = titles[i]                       # << título correspondiente
        axes[idx].set_title(f"{title} Threshold") # ← ¡aquí el formato!
        #axes[idx].set_title(f'{metodo} ({sentinel_id})')
        axes[idx].set_xlabel("Longitud")
        axes[idx].set_ylabel("Latitud")
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if extent:
            axes[idx].set_xlim(extent[0], extent[1])
            axes[idx].set_ylim(extent[2], extent[3])

        axes[idx].text(
            0.02, 0.02,
            f"Threshold = {threshold_val:.4f}",
            transform=axes[idx].transAxes,
            fontsize=10,
            color='black',
            verticalalignment='bottom',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.7)
        )
        idx += 1

    # --- Apagar subplots vacíos
    for j in range(idx, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()

    plt.savefig(os.path.join(output_directory, f"{sentinel_id}_ThresholdsCompare.png"), dpi=300)
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    
    
    
def histograma_threshold_metrics(
    glcm,
    window_size=31,
    k=0.2,
    r=None,
    bins=256,
    figsize=(10, 6)
):
    """
    Calcula thresholds (Otsu, Niblack, Sauvola, Li) sobre `glcm` y
    muestra un histograma con las líneas de cada umbral y su métrica
    interna (varianza inter-clase / varianza total).

    Parámetros
    ----------
    glcm : ndarray
        Imagen (2-D) con intensidad de textura. Puede contener NaN.
    window_size, k, r : parámetros de Niblack y Sauvola.
    bins : int
        Número de clases del histograma.
    figsize : tuple
        Tamaño de la figura en pulgadas.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.filters import (
        threshold_otsu, threshold_niblack,
        threshold_sauvola, threshold_li
    )

    # --- 1. Pre-procesar: aplanar y descartar NaN ---
    data = glcm.astype(np.float64).ravel()
    data = data[~np.isnan(data)]

    # --- 2. Thresholds ---
    thr_otsu    = threshold_otsu   (data)
    thr_niblack = threshold_niblack(data.reshape(glcm.shape),
                                    window_size=window_size, k=k).mean()
    thr_sauvola = threshold_sauvola(data.reshape(glcm.shape),
                                    window_size=window_size, k=k, r=r).mean()
    thr_li      = threshold_li    (data)

    thresholds = {
        "Otsu"   : thr_otsu,
        "Niblack": thr_niblack,
        "Sauvola": thr_sauvola,
        "Li"     : thr_li
    }

    # --- 3. Métrica interna: varianza inter-clase / total ---
    def calidad(img, thr):
        """Between-Class Variance ÷ Total Variance (máx=1)."""
        if img.size == 0: return 0.0
        mu_T  = img.mean()
        var_T = img.var() + 1e-9
        mask0 = img <= thr
        mask1 = ~mask0
        w0, w1 = mask0.mean(), mask1.mean()
        if w0 == 0 or w1 == 0:
            return 0.0
        mu0, mu1 = img[mask0].mean(), img[mask1].mean()
        bc_var = w0 * (mu0 - mu_T) ** 2 + w1 * (mu1 - mu_T) ** 2
        return bc_var / var_T

    scores = {m: calidad(data, t) for m, t in thresholds.items()}

    # --- 4. Plot ---
    plt.figure(figsize=figsize)
    n, bins_edges, _ = plt.hist(
        data,
        bins=bins,
        color="lightgray",
        alpha=0.65,
        edgecolor="none",
        label="Histograma GLCM"
    )

    colores = {
        "Otsu":    "tab:blue",
        "Niblack": "tab:orange",
        "Sauvola": "tab:green",
        "Li":      "tab:red"
    }

    for metodo, thr in thresholds.items():
        plt.axvline(
            thr,
            color=colores[metodo],
            linewidth=2,
            linestyle="--",
            label=f"{metodo}: {thr:.3f}  |  Score: {scores[metodo]:.3f}"
        )

    plt.title("Histogram of GLCM values\n+ thresholds & internal quality score")
    plt.xlabel("Valor GLCM")
    plt.ylabel("Frecuencia")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # --- 5. Devolver thresholds y scores por si los necesitas ---
    return thresholds, scores





def n_lineabase_dsas(
    productos_binarios,
    ground_truth_path,
    output_directory,
    sentinel_id=""
):
    """
    Calcula métricas de desempeño de cada método de umbralización
    comparándolas con una máscara ground-truth y las grafica.

    Parámetros
    ----------
    productos_binarios : dict
        {'flood_otsu': Product, 'flood_niblack': Product, ...}
    ground_truth_path : str
        Ruta al raster ground-truth (1 = agua, 0 = tierra, -1 = nodata).
    output_directory : str
        Carpeta donde se guardarán las figuras.
    sentinel_id : str
        Identificador opcional del producto Sentinel (para títulos).
    """
    import numpy as np
    import os
    import rasterio
    from sklearn.metrics import (
        confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score
    )
    import pandas as pd
    import coastalcf.n_lineabase_dsas as wb
    from matplotlib import pyplot as plt

    # --- 1. CARGAR GROUND-TRUTH EN UN ARRAY ---
    with rasterio.open(ground_truth_path) as gt_ds:
        gt_mask = gt_ds.read(1)  # asume single-band
    gt_valid = gt_mask != -1    # evita nodata
    y_true  = gt_mask[gt_valid].astype(np.uint8)

    # --- 2. RECORRER CADA MÉTODO Y CALCULAR MÉTRICAS ---
    resultados = []   # lista de dicts → DataFrame

    for nombre, producto in productos_binarios.items():
        # BandMaths de cada entrada tiene UNA sola banda
        band_name = [b for b in producto.getBandNames()][0]
        band = producto.getBand(band_name)
        w, h = band.getRasterWidth(), band.getRasterHeight()
        data = np.zeros(w * h, np.float32)
        band.readPixels(0, 0, w, h, data)
        y_pred_full = data.reshape((h, w))

        # Ajustar tamaño a GT (solo si son idénticas dims):
        y_pred = y_pred_full[gt_valid].astype(np.uint8)

        # Confusión-matrix y métricas
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        acc  = accuracy_score (y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score   (y_true, y_pred, zero_division=0)
        f1   = f1_score      (y_true, y_pred, zero_division=0)

        resultados.append({
            "Método": nombre.replace("flood_", "").capitalize(),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })

    df = pd.DataFrame(resultados).set_index("Método")
    print("\n=== Métricas ===\n", df.round(3))

    # --- 3. GRAFICAR BARRAS DE MÉTRICA PRINCIPAL (F1) ---
    wb.bar(
        df.reset_index(),
        x="Método", y="F1",
        title=f"F1-score por método ({sentinel_id})",
        ylabel="F1",
        ylim=(0, 1)
    )

    # También puedes agregar Accuracy, Precision, Recall:
    # wb.line(df.reset_index(), x="Método", y=["Accuracy","Precision","Recall","F1"])

    # --- 4. GUARDAR FIGURA ---
    plt.savefig(os.path.join(output_directory, f"{sentinel_id}_performance.png"), dpi=300)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # --- 5. Devolver DataFrame si quieres seguir usándolo ---
    return df

