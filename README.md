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

<p align="center">
  <img width="9696" height="1147" alt="Análisis ENSO 2023-2024 (14)" src="https://github.com/user-attachments/assets/2d592c57-b4f3-4339-8ea0-80b0034fa222" />
</p>

## Módulos principales

La funcionalidad se organiza en los siguientes métodos disponibles en el paquete:

```python
__all__ = [
    "a_sentinel1_id",
    "a_lectura_metadata",
    "b_ruido_termico",
    "c_calibracion_radiometrica",
    "d_submuestreo",
    "e_operador_multilook",
    "f_filtro_speckle",
    "g_nivelacion_terreno",
    "h_glcm",
    "h_exportar_sentinel1",
    "i_binarizacion",
    "j_correcion_geometrica",
    "k_graficos_binarizacion",
    "l_raster_a_shapefile",
    "m_suavizar_shapefile",
    "n_lineabase_dsas",
    "o_limpieza_lineacosta",
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

