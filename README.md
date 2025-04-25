# Sentinel-1 Coastline Delineation with Snappy

Este repositorio contiene un ejemplo completo de procesamiento de imágenes Sentinel-1 utilizando **esa_snappy** (la API de Python para SNAP), enfocado en la **detección de línea de costa** a partir de imágenes SAR. El flujo de trabajo se basa en experiencias previas y ha sido adaptado y extendido para facilitar su aplicación en investigaciones científicas y proyectos de monitoreo ambiental.

## 🛰️ Datos de entrada

- **Satélite:** Sentinel-1B
- **Producto:** GRD descomprimido (modo IW, polarización VH)
- **Archivo de ejemplo:** `S1B_IW_GRDH_1SDV_20180219T002238_20180219T002303_009685_011789_DEDC`

##  Flujo de procesamiento

1. Carga y recorte del área de estudio  
2. Eliminación de ruido térmico  
3. Calibración radiométrica  
4. Aplicación de multilooking  
5. Filtrado de speckle  
6. Nivelación del terreno (Terrain Flattening)  
7. Cálculo de texturas GLCM  
8. Corrección geométrica (Terrain Correction)  
9. Umbralización (Sauvola u otros métodos)  
10. Creación de rutas y máscara binaria  
11. Exportación del raster y shapefile final

![image](https://github.com/user-attachments/assets/d42e1b0d-32a4-4cd2-aea7-4b60d0e23ff3)

## Módulos principales

La funcionalidad se organiza en los siguientes métodos disponibles en el paquete:

```python
__all__ = [
    "readMetadata",
    "do_thermal_noise_removal",
    "radiometricCalibration",
    "subset",
    "perform_multilook",
    "speckleFiltering",
    "perform_terrain_flattening",
    "glcmOp",
    "waterDetectionBinarization",
    "geometricCorrection",
    "generate_raster_path",
    "generate_shapefile_path",
    "exportar_raster_y_shapefile",
    "get_sentinel_fecha_id",
    "plot_binary_map",
    "plot_histogram",
    "plot_thresholds",
    "WDBThreshold"
]
```

## Referencias y créditos

Adaptado de:
- [Pastebin script (2017)](https://pastebin.com/dU4AUr3B)
- [Sentinel-1 preprocessing using Snappy](https://github.com/wajuqi/Sentinel-1-preprocessing-using-Snappy)
- [EVER-EST WP5 Deliverable D5.9](https://ever-est.eu/wp-content/uploads/EVER-EST-DEL-WP5-D5.9.pdf)

---

###  Autores

Este trabajo fue elaborado por:

- **Cristopher Enrique Foster Velázquez**  
- **Alejandra López Caloca**  
- **Fernando Lopez Caloca**

---

## Licencia

Este proyecto se comparte bajo la licencia GNU GENERAL PUBLIC LICENSE
```

