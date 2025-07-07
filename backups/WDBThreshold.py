# WDBThreshold.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage.filters import threshold_niblack, threshold_sauvola

def plot_histogramas_con_umbral(img_db_valid, thresholds, titles, sentinel_id, output_dir, save_plots=False):
    """
    Grafica histogramas con curvas KDE suavizadas y líneas de umbral para distintos métodos de umbralización.

    Args:
        img_db_valid (np.ndarray): Imagen en dB aplanada.
        thresholds (list): Lista de valores de umbral por método.
        titles (list): Lista de títulos de los métodos.
        sentinel_id (str): ID del producto Sentinel (para el nombre del archivo).
        output_dir (str): Carpeta de salida para guardar la figura.
        save_plots (bool): Si se guarda o no la figura.
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 18))
    axes = axes.flatten()

    # Histograma general
    counts_0, bins_0, _ = axes[0].hist(img_db_valid.ravel(), bins=300, range=[-5, 20], color='blue', alpha=0.8) #512
    total_pixels_0 = np.sum(counts_0)
    bin_centers_0 = 0.5 * (bins_0[:-1] + bins_0[1:])
    kde_0 = gaussian_kde(img_db_valid.ravel())
    kde_values_0 = kde_0(bin_centers_0)
    kde_scaled_0 = kde_values_0 * total_pixels_0 * (bins_0[1] - bins_0[0])
    axes[0].plot(bin_centers_0, kde_scaled_0, color='black', linewidth=3.5)#, label='Distribución suavizada')

    #axes[0].set_title('Histograma VH en dB', fontsize=14)
    axes[0].set_title('Gamma0 VH band Histogram', fontsize=14)
    #axes[0].set_xlabel('Valor (dB)', fontsize=12)
    axes[0].set_xlabel('dB', fontsize=12)
    #axes[0].set_ylabel('Frecuencia', fontsize=12)
    axes[0].set_ylabel('Frecuency', fontsize=12)
    axes[0].set_ylim(0, 6000)
    axes[0].legend(loc='lower left', fontsize=10)

    # Histogramas con umbrales
    for idx, (thresh, title) in enumerate(zip(thresholds, titles)):
        counts, bins, patches = axes[idx+1].hist(
            img_db_valid.ravel(), bins=300, range=[-5, 20], color='gray', alpha=0.8)
        total_pixels = np.sum(counts)

        # Colorear los bins según el umbral
        for bin_left, patch in zip(bins[:-1], patches):
            patch.set_facecolor('darkorange' if bin_left < thresh else 'darkblue')

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        kde = gaussian_kde(img_db_valid.ravel())
        kde_values = kde(bin_centers)
        kde_scaled = kde_values * total_pixels * (bins[1] - bins[0])
        axes[idx+1].plot(bin_centers, kde_scaled, color='black', linewidth=3.5)#, label='Distribución suavizada')

        # Línea vertical del umbral
        #axes[idx+1].axvline(thresh, color='red', linestyle='--', label=f'Umbral: {thresh:.2f} dB')
        axes[idx+1].axvline(thresh, color='red', linestyle='--', label=f'Threshold:\n{thresh:.2f} dB')

        # Porcentajes
        below_thresh = np.sum(counts[bins[:-1] < thresh])
        above_thresh = total_pixels - below_thresh
        perc_below = (below_thresh / total_pixels) * 100
        perc_above = (above_thresh / total_pixels) * 100

        axes[idx+1].text(0.05, 0.95, f"> {thresh:.2f} dB:\n{perc_above:.1f}%",
                         transform=axes[idx+1].transAxes, fontsize=10,
                         verticalalignment='top', color='darkorange')
        axes[idx+1].text(0.80, 0.95, f"< {thresh:.2f} dB:\n{perc_below:.1f}%",
                         transform=axes[idx+1].transAxes, fontsize=10,
                         verticalalignment='top', color='darkblue')

        # Mejoras visuales
        title1=title+" Threshold"
        axes[idx+1].set_title(title1, fontsize=16)
        #axes[idx+1].set_xlabel('Valor (dB)', fontsize=14)
        axes[idx+1].set_xlabel('Gamma0 VH band (dB)', fontsize=12)
        #axes[idx+1].set_ylabel('Frecuencia', fontsize=14)
        axes[idx+1].set_ylabel('Frecuency', fontsize=12)
        axes[idx+1].set_ylim(0, 6000)
        axes[idx+1].legend(loc='center right', fontsize=10)

    plt.tight_layout()
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{sentinel_id}_histograms.png"), dpi=300)
    plt.show()
    plt.pause(2)
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from skimage.filters import (
    threshold_otsu,
    threshold_niblack,
    threshold_sauvola,
    threshold_li,
)

def _cross_entropy(image, t):
    foreground = image > t
    background = ~foreground
    mean_fore = np.mean(image[foreground]) if np.any(foreground) else 1e-6
    mean_back = np.mean(image[background]) if np.any(background) else 1e-6
    mean_fore = np.maximum(mean_fore, 1e-6)
    mean_back = np.maximum(mean_back, 1e-6)
    return - (np.sum(np.log(mean_fore) * image[foreground]) + np.sum(np.log(mean_back) * image[background]))

def li_gradient(image, t):
    foreground = image > t
    background = ~foreground
    if np.any(foreground) and np.any(background):
        mean_fore = np.mean(image[foreground])
        mean_back = np.mean(image[background])
        if mean_fore > 0 and mean_back > 0:
            return (mean_back - mean_fore) / (np.log(mean_back) - np.log(mean_fore)) - t
    return 0.0

def li_iterative_threshold(image, extra_initial_value=None):
    li_val = threshold_li(image)
    li_iter_val = li_val + 5
    li_iter_history = [li_iter_val]
    max_iter = 20
    min_iter = 5
    tol = 1e-6
    for i in range(max_iter):
        dt = li_gradient(image, li_iter_val)
        new_val = li_iter_val + dt
        li_iter_history.append(new_val)
        if i >= min_iter and abs(dt) < tol:
            break
        li_iter_val = new_val
    return li_val, li_iter_val, li_iter_history

def plot_threshold_histograms_extended(img_db_valid, thresholds, titles):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde
    from matplotlib import cm
    from skimage.filters import threshold_sauvola, threshold_niblack

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    # 1. Iterative Li Threshold Optimization
    li_val, li_iter_val, li_iter_history = li_iterative_threshold(img_db_valid, extra_initial_value=10)
    thresholds_all = np.linspace(np.min(img_db_valid), np.max(img_db_valid), 512)
    entropies_iter = [_cross_entropy(img_db_valid, t) for t in li_iter_history]
    entropies_all = [_cross_entropy(img_db_valid, t) for t in thresholds_all]
    axes[0].plot(thresholds_all, entropies_all, color='blue', label='Cross Entropy')
    axes[0].plot(li_iter_history, entropies_iter, marker='o', color='maroon', label='Iterative Li')
    axes[0].axvline(li_val, color='red', linestyle='--', label='Original Li')
    axes[0].set_title("Iterative Li Optimization (Cross Entropy)")
    axes[0].set_xlabel("Gamma0 VH band (dB)")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Otsu Threshold Variance Analysis
    thresholds_range = np.linspace(np.min(img_db_valid), np.max(img_db_valid), 512)
    variances = []
    for t in thresholds_range:
        background = img_db_valid[img_db_valid <= t]
        foreground = img_db_valid[img_db_valid > t]
        w_b = len(background) / len(img_db_valid.ravel())
        w_f = len(foreground) / len(img_db_valid.ravel())
        var_b = np.var(background) if len(background) > 0 else 0
        var_f = np.var(foreground) if len(foreground) > 0 else 0
        variances.append(w_b * var_b + w_f * var_f)
    otsu_val = thresholds[0]
    axes[1].plot(thresholds_range, variances, color='blue')
    axes[1].axvline(otsu_val, color='red', linestyle='--', label='Otsu Threshold')
    axes[1].set_title("Intra-class Variance vs Threshold (Otsu)")
    axes[1].set_xlabel("Gamma0 VH band (dB)")
    axes[1].set_ylabel("Intra-class Variance")
    axes[1].legend()
    axes[1].grid(True)

    # 3. Sauvola Threshold Map
    sauvola_thresh = threshold_sauvola(img_db_valid, window_size=15, k=0.2)
    im2 = axes[2].imshow(sauvola_thresh, cmap='viridis')
    axes[2].set_title("Local Threshold Map - Sauvola")
    fig.colorbar(im2, ax=axes[2], shrink=0.6)

    # 4. Niblack Threshold Map
    niblack_thresh = threshold_niblack(img_db_valid, window_size=15, k=0.2)
    im3 = axes[3].imshow(niblack_thresh, cmap='viridis')
    axes[3].set_title("Local Threshold Map - Niblack")
    fig.colorbar(im3, ax=axes[3], shrink=0.6)

    # 5. Histogram of Local Thresholds
    axes[4].hist(sauvola_thresh.ravel(), bins=256, alpha=0.6, label='Sauvola', color='darkred')
    axes[4].hist(niblack_thresh.ravel(), bins=256, alpha=0.6, label='Niblack', color='darkorange')
    axes[4].set_ylim(0, 6000)
    axes[4].set_xlabel("Gamma0 VH band (dB)",fontsize=10)
    axes[4].set_ylabel("Frequency",fontsize=10)
    axes[4].set_title("Histogram of Local Thresholds")
    axes[4].legend()
    axes[4].grid(True)

    # 6. Global Histogram with KDE and Only Sauvola/Niblack Thresholds
    counts, bins, _ = axes[5].hist(img_db_valid.ravel(), bins=300, range=[-5, 20], color='midnightblue', alpha=0.7)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    kde = gaussian_kde(img_db_valid.ravel())
    kde_values = kde(bin_centers)
    kde_scaled = kde_values * np.sum(counts) * (bins[1] - bins[0])
    axes[5].plot(bin_centers, kde_scaled, color='black', linewidth=2.5, label='KDE')

    # Only draw vertical lines for Sauvola and Niblack
    try:
        sauvola_val = thresholds[titles.index('Sauvola')]
        niblack_val = thresholds[titles.index('Niblack')]
        axes[5].axvline(sauvola_val, linestyle='--', color='darkred', linewidth=2, label='Sauvola Threshold')
        axes[5].axvline(niblack_val, linestyle='--', color='darkorange', linewidth=2, label='Niblack Threshold')
    except ValueError:
        print("Warning: 'Sauvola' or 'Niblack' not found in titles list.")

    axes[5].set_title("Global VH Histogram with Selected Thresholds")
    axes[5].set_xlabel("Gamma0 VH band (dB)",fontsize=10)
    axes[5].set_ylabel("Frequency",fontsize=10)
    axes[5].set_ylim(0, 6000)
    axes[5].legend(fontsize=8)
    axes[5].grid(True)

    plt.tight_layout()
    return fig




def WDBThreshold(
    textura,
    sentinel_1_path,
    window_size=15,
    k=0.2,
    r=None,
    save_plots=True,
    output_dir=None
    ):
    
    from esa_snappy import GPF, HashMap, jpy, Product
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola, threshold_li
    from .get_sentinel_fecha_id import get_sentinel_fecha_id
    import os
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    from scipy.stats import gaussian_kde


    sentinel_id = get_sentinel_fecha_id(sentinel_1_path)
    print("\n--- Iniciando detección de agua con TODOS los métodos ---")

    available_bands = list(textura.getBandNames())
    print(f"Bandas disponibles: {available_bands}")
    if 'Gamma0_VH_GLCMMean' not in available_bands:
        raise RuntimeError("La banda 'Gamma0_VH_GLCMMean' no está disponible en el producto.")

    paramToDB = HashMap()
    paramToDB.put('sourceBands', 'Gamma0_VH_GLCMMean')

    band = textura.getBand("Gamma0_VH_GLCMMean")
    data = np.zeros(band.getRasterWidth() * band.getRasterHeight(), np.float32)
    band.readPixels(0, 0, band.getRasterWidth(), band.getRasterHeight(), data)
    print("Rango de datos lineales:", np.min(data), np.max(data))
    
    # --- Filtrar datos inválidos en ESCALA LINEAL ---
    data = data.reshape((band.getRasterHeight(), band.getRasterWidth()))
    data_masked = np.where(data <= 0, np.nan, data)  # <=0 es el corte físico correcto

    # Opcional: si quieres forzar que los NaN sean -9999 para compatibilidad
    data_valid = np.nan_to_num(data_masked, nan=-9999)
    print("Rango de datos lineales:", np.min(data), np.max(data))
    # --- Convertir a escala de dB solo los datos válidos ---
    try:
        print("Convirtiendo a escala de decibeles (dB)...")
        paramToDB = HashMap()
        paramToDB.put('sourceBands', 'Gamma0_VH_GLCMMean')
        producto_db = GPF.createProduct("LinearToFromdB", paramToDB, textura)
        banda_db = producto_db.getBand('Gamma0_VH_GLCMMean_db')
        print("Conversión a dB completada.")
    except Exception as e:
        raise RuntimeError("Error al convertir la banda a dB.") from e


    w, h = banda_db.getRasterWidth(), banda_db.getRasterHeight()
    img_db_data = np.zeros(w * h, np.float32)
    banda_db.readPixels(0, 0, w, h, img_db_data)

    min_val, max_val = np.min(img_db_data), np.max(img_db_data)
    print(f"Rango de valores en dB: Min={min_val:.2f}, Max={max_val:.2f}")

    img_db_data = img_db_data.reshape((h, w))
    img_db_valid = img_db_data

    print("Calculando umbrales (Otsu, Niblack, Sauvola, Li)...")
    otsu_val = threshold_otsu(img_db_valid)
    niblack_val = np.mean(threshold_niblack(img_db_valid, window_size=window_size, k=k))
    sauvola_val = np.mean(threshold_sauvola(img_db_valid, window_size=window_size))
    #li_val = threshold_li(img_db_valid)


    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from matplotlib import patches
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    from skimage.filters import threshold_li

    def _cross_entropy(image, t):
        """Calcula la cross-entropy para un umbral dado."""
        foreground = image > t
        background = ~foreground

        mean_fore = np.mean(image[foreground]) if np.any(foreground) else 1e-6
        mean_back = np.mean(image[background]) if np.any(background) else 1e-6

        # Protecciones extra contra log(0) o negativos
        mean_fore = np.maximum(mean_fore, 1e-6)
        mean_back = np.maximum(mean_back, 1e-6)

        cross_entropy = - (np.sum(np.log(mean_fore) * image[foreground]) + np.sum(np.log(mean_back) * image[background]))
        return cross_entropy

    def li_gradient(image, t):
        """Calcula la actualización de umbral en Li."""
        foreground = image > t
        background = ~foreground

        if np.any(foreground) and np.any(background):
            mean_fore = np.mean(image[foreground])
            mean_back = np.mean(image[background])

            if mean_fore > 0 and mean_back > 0:
                t_next = (mean_back - mean_fore) / (np.log(mean_back) - np.log(mean_fore))
                dt = t_next - t
                return dt
        return 0.0
    # --- Método Li Iterativo ---
    
    def li_iterative_threshold(
        img_db_valid, 
        save_plots=False, 
        output_dir="plots", 
        sentinel_id="sentinel",
        extra_initial_value=None
    ):
        # 1. Threshold inicial de Li
        li_val = threshold_li(img_db_valid)
    
        # 2. Generar todos los thresholds posibles en el rango
        thresholds_all = np.linspace(np.min(img_db_valid), np.max(img_db_valid), 512)
        entropies_all = np.array([_cross_entropy(img_db_valid, t) for t in thresholds_all])

        # 3. Optimización partiendo de li_val
        li_iter_val = li_val + 5#- 1
        li_iter_history = [li_iter_val]

        max_iter = 20
        min_iter = 5
        tol = 1e-6

        for i in range(max_iter):
            dt = li_gradient(img_db_valid, li_iter_val)
            new_val = li_iter_val + dt
            li_iter_history.append(new_val)

            if i >= min_iter and abs(dt) < tol:
                break
            li_iter_val = new_val

        entropies_iter = np.array([_cross_entropy(img_db_valid, t) for t in li_iter_history])

        # 4. Optimización partiendo de otro valor (opcional)
        if extra_initial_value is not None:
            extra_iter_val = extra_initial_value
            extra_iter_history = [extra_iter_val]

            for i in range(max_iter):
                dt_extra = li_gradient(img_db_valid, extra_iter_val)
                new_val_extra = extra_iter_val + dt_extra
                extra_iter_history.append(new_val_extra)

                if i >= min_iter and abs(dt_extra) < tol:
                    break
                extra_iter_val = new_val_extra

            entropies_extra_iter = np.array([_cross_entropy(img_db_valid, t) for t in extra_iter_history])
        else:
            extra_iter_history = None
            entropies_extra_iter = None

        # 5. --- Gráfica principal ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # --- Toda la entropía
        ax.plot(thresholds_all, entropies_all, label='All threshold entropies', color='C0')

    
        # --- Path optimización alternativo (si existe)
        if extra_iter_history is not None:
            ax.plot(extra_iter_history, entropies_extra_iter, marker='^', linestyle='--', color='C2', label='Optimization path (Demostrative)')
            ax.scatter(extra_iter_history, entropies_extra_iter, color='C2')
            
        # --- Path optimización estándar
        ax.plot(li_iter_history, entropies_iter, marker='o', linestyle='-', color='C1', label='Optimization path (Li)')
        ax.scatter(li_iter_history, entropies_iter, color='C1')
        
        # Calcular Otsu y su entropía cruzada
        otsu_val = threshold_otsu(img_db_valid)
        otsu_entropy = _cross_entropy(img_db_valid, otsu_val)
        # Agregar al gráfico principal
        ax.axvline(otsu_val, color='purple', linestyle=':', label=f'Otsu: {otsu_val:.2f} dB')
        ax.scatter([otsu_val], [otsu_entropy], color='purple', marker='x', s=80)



        #ax.set_title('Proceso de Optimización de Li Iterativo (Cross Entropy)', fontsize=14)
        ax.set_title('Optimization path: Iterative Li (Cross Entropy)', fontsize=14)
        ax.set_xlabel('Threshold (dB)', fontsize=12)
        ax.set_ylabel('Cross Entropy', fontsize=12)
        ax.legend()
        ax.grid(True)

        # --- Inset (zoom) ---
        axins = inset_axes(ax, width="35%", height="35%", loc='center right', borderpad=2)

        axins.plot(thresholds_all, entropies_all, color='C0')

        if extra_iter_history is not None:
            axins.plot(extra_iter_history, entropies_extra_iter, marker='^', linestyle='--', color='C2')
            axins.scatter(extra_iter_history, entropies_extra_iter, color='C2')
            # Agregar también al inset
            axins.axvline(otsu_val, color='purple', linestyle=':')
            axins.scatter([otsu_val], [otsu_entropy], color='purple', marker='x', s=60)

        axins.plot(li_iter_history, entropies_iter, marker='o', linestyle='-', color='C1')
        axins.scatter(li_iter_history, entropies_iter, color='C1')
        
        x_center = li_iter_val
        y_center = _cross_entropy(img_db_valid, li_iter_val)
        
        if not np.isfinite(x_center) or not np.isfinite(y_center):
            x_center = np.nanmean(thresholds_all)
            y_center = np.nanmean(entropies_all)

        x_width = 0.02
        y_width = 0.02 * (np.nanmax(entropies_all) - np.nanmin(entropies_all))

        if not np.isfinite(x_width) or x_width == 0:
            x_width = 0.01
        if not np.isfinite(y_width) or y_width == 0:
            y_width = 0.01

        axins.set_xlim(x_center - x_width, x_center + x_width)
        axins.set_ylim(y_center - y_width, y_center + y_width)
        axins.set_xticks([])
        axins.set_yticks([])

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, f"{sentinel_id}_li_iterativo_crossentropy.png"), dpi=300)

        plt.show()
        plt.pause(2)
        plt.close()

        return li_val, li_iter_val, li_iter_history


    # Con otro valor inicial (por ejemplo, 1 unidad arriba de li_val)
    li_val, li_iter_val, li_iter_history = li_iterative_threshold(
        img_db_valid,
        extra_initial_value= 3.5
    )


    print(f"Threshold inicial de Li: {li_val}")
    print(f"Threshold optimizado de Li Iterativo: {li_iter_val}")


    print(f"  - Otsu         = {otsu_val:.3f}")
    print(f"  - Niblack      = {niblack_val:.3f}")
    print(f"  - Sauvola      = {sauvola_val:.3f}")
    print(f"  - Li           = {li_val:.3f}")
    print(f"  - Li iterativo = {li_iter_val:.3f}")

    def dB_to_linear(db_val):
        return math.pow(10, (db_val / 10))

    pow_otsu = dB_to_linear(otsu_val)
    pow_niblack = dB_to_linear(niblack_val)
    pow_sauvola = dB_to_linear(sauvola_val)
    pow_li = dB_to_linear(li_val)
    pow_li_iter = dB_to_linear(li_iter_val)

    expr_otsu    = f"(Gamma0_VH_GLCMMean < {pow_otsu}) ? 0 : 1"
    expr_niblack = f"(Gamma0_VH_GLCMMean < {pow_niblack}) ? 0 : 1"
    expr_sauvola = f"(Gamma0_VH_GLCMMean < {pow_sauvola}) ? 0 : 1"
    expr_li      = f"(Gamma0_VH_GLCMMean < {pow_li}) ? 0 : 1"
    expr_li_iter = f"(Gamma0_VH_GLCMMean < {pow_li_iter}) ? 0 : 1"

    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')

    def crear_band_descriptor(nombre, expresion):
        bd = BandDescriptor()
        bd.name = nombre
        bd.type = 'Int32'
        bd.expression = expresion
        return bd

    bd_otsu = crear_band_descriptor('flood_otsu', expr_otsu)
    bd_niblack = crear_band_descriptor('flood_niblack', expr_niblack)
    bd_sauvola = crear_band_descriptor('flood_sauvola', expr_sauvola)
    bd_li = crear_band_descriptor('flood_li', expr_li)
    bd_li_iter = crear_band_descriptor('flood_li_iter', expr_li_iter)

    targetBands = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 5)
    targetBands[0] = bd_otsu
    targetBands[1] = bd_niblack
    targetBands[2] = bd_sauvola
    targetBands[3] = bd_li
    targetBands[4] = bd_li_iter

    parameters = HashMap()
    parameters.put('targetBands', targetBands)

    productos_binarios = {}
    descriptores = {
        "flood_otsu": bd_otsu,
        "flood_niblack": bd_niblack,
        "flood_sauvola": bd_sauvola,
        "flood_li": bd_li,
        "flood_li_iter": bd_li_iter
    }

    for nombre, descriptor in descriptores.items():
        print(f"Creando producto individual para: {nombre}")
        targetBandArray = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
        targetBandArray[0] = descriptor
        params = HashMap()
        params.put('targetBands', targetBandArray)

        try:
            producto = GPF.createProduct('BandMaths', params, textura)
            productos_binarios[nombre] = producto
            print(f"Producto creado: {nombre}")
        except Exception as e:
            raise RuntimeError(f"Error al crear la máscara {nombre} con BandMaths.") from e

    print("\nCreando producto SNAP para img_db_valid...")

    # Crear el descriptor de banda para img_db_valid
    BandDescriptor = jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')

    bd_img_db_valid = BandDescriptor()
    bd_img_db_valid.name = 'img_db_valid'
    bd_img_db_valid.type = 'Float32'
    bd_img_db_valid.expression = 'Gamma0_VH_GLCMMean_db'  # 🔥 Usa la banda ya convertida a dB

    # Crear el arreglo de bandas (sólo una banda)
    targetBandArray = jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
    targetBandArray[0] = bd_img_db_valid

    # Parámetros
    params = HashMap()
    params.put('targetBands', targetBandArray)

    # Crear el producto usando GPF (como las máscaras)
    try:
        img_db_valid_product = GPF.createProduct('BandMaths', params, producto_db)
        print("Producto SNAP para img_db_valid creado exitosamente.")
    except Exception as e:
        raise RuntimeError("Error al crear el producto img_db_valid con BandMaths.") from e

    li_val, li_iter_val, li_iter_history = li_iterative_threshold(img_db_valid, extra_initial_value=3.5)
    
    plot_histogramas_con_umbral(
    img_db_valid=img_db_valid,
    thresholds=[otsu_val, niblack_val, sauvola_val, li_val, li_iter_val],
    titles=['Otsu', 'Niblack', 'Sauvola', 'Li', 'Iterative Li'],
    sentinel_id=sentinel_id,
    output_dir=output_dir,
    save_plots=True
    )

    thresholds = [otsu_val, niblack_val, sauvola_val, li_val, li_iter_val]
    titles = ['Otsu', 'Niblack', 'Sauvola', 'Li', 'Li Iterativo']
    fig = plot_threshold_histograms_extended(img_db_valid, thresholds, titles)
    #plt.suptitle("Visualizaciones Comparativas de Umbrales", fontsize=18, y=1.02)
    plt.suptitle("Comparative Threshold Visualization", fontsize=18, y=1.02)
    plt.show()

    return productos_binarios, {
        "otsu": otsu_val,
        "niblack": niblack_val,
        "sauvola": sauvola_val,
        "li": li_val,
        "li_iterativo": li_iter_val
    }, img_db_valid_product  
