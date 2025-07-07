# Cristopher DSAS Pipeline PRO v2.0 (Modular)

import os
import geopandas as gpd
import pandas as pd
from shapely.ops import linemerge, unary_union
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL

# Directorio raíz
base_dir = r'C:\Users\c4cfo\OneDrive\CENTROGEO-cfoster\1_ENSOCE\C3_Output\shapes'

# Shapefile de recorte
clip_shapefile = os.path.join(base_dir, 'coastlinesh4326.shp')

# Años que deseas procesar
years = ["2022", "2023", "2024"]

# Diccionario de meses
meses_dict = {
    "1_enero": "a_enero",
    "2_febrero": "b_febrero",
    "3_marzo": "c_marzo",
    "4_abril": "d_abril",
    "5_mayo": "e_mayo",
    "6_junio": "f_junio",
    "7_julio": "g_julio",
    "8_agosto": "h_agosto",
    "9_septiembre": "i_septiembre",
    "10_octubre": "j_octubre",
    "11_noviembre": "k_noviembre",
    "12_diciembre": "l_diciembre"
}


# ─────────────────────────────────────────────────────────────────────
# FUNCIONES DEL PIPELINE

def procesar_anio(input_base, output_base, clip_gdf, meses_id):
    """Procesa un solo año y devuelve los GeoDataFrames procesados"""
    gdfs_clipped = []

    for subfolder in os.listdir(input_base):
        subfolder_path = os.path.join(input_base, subfolder)
        if os.path.isdir(subfolder_path):
            shp_files = [f for f in os.listdir(subfolder_path) if f.endswith('.shp')]
            if not shp_files:
                continue

            shp_path = os.path.join(subfolder_path, shp_files[0])
            gdf = gpd.read_file(shp_path).to_crs(epsg=32613)
            clipped = gpd.overlay(gdf, clip_gdf, how='intersection')

            if not clipped.empty:
                exploded = clipped.explode(index_parts=False)
                exploded = exploded[exploded.length > 80000]

                if not exploded.empty:
                    try:
                        if len(exploded) > 1:
                            buffered = exploded.buffer(100)
                            unioned = unary_union(buffered)

                            if unioned.geom_type == "Polygon":
                                outline = LineString(unioned.exterior)
                            elif unioned.geom_type == "MultiPolygon":
                                largest = max(unioned.geoms, key=lambda p: p.area)
                                outline = LineString(largest.exterior)
                            else:
                                outline = unioned
                        else:
                            outline = exploded.geometry.values[0]

                        merged_line = outline if isinstance(outline, LineString) else linemerge(outline)
                        if merged_line.geom_type == 'MultiLineString':
                            merged_line = max(merged_line.geoms, key=lambda x: x.length)

                        id_label = meses_id.get(subfolder, subfolder)
                        longest_line = gpd.GeoDataFrame({'id': [id_label]}, geometry=[merged_line], crs="EPSG:32613")

                        output_name = f"{subfolder}_mz.shp"
                        output_path = os.path.join(output_base, output_name)
                        longest_line.to_file(output_path)
                        gdfs_clipped.append(longest_line)

                    except Exception as e:
                        print(f"Error procesando {subfolder}: {e}")

    return gdfs_clipped


def limpiar_unir_geodfs(gdfs_list):
    """Unifica y limpia todos los GeoDataFrames procesados"""
    merged = gpd.GeoDataFrame(pd.concat(gdfs_list, ignore_index=True), crs="EPSG:32613")

    # Limpieza topológica
    def force_linestring(geom):
        if geom.geom_type == 'MultiLineString':
            return max(list(geom.geoms), key=lambda x: x.length)
        elif geom.geom_type == 'LineString':
            return geom
        return None

    def strip_z(geom):
        if hasattr(geom, 'coords'):
            coords_2d = [(x, y) for x, y, *_ in geom.coords]
            return LineString(coords_2d)
        return geom

    merged['geometry'] = merged['geometry'].apply(force_linestring).apply(strip_z)
    merged = merged[merged.is_valid]
    return merged


def generar_linea_base(shorelines_gdf, buffer_distance=-0.0002):
    """Genera automáticamente la línea base DSAS"""
    union_all = unary_union(shorelines_gdf.geometry)
    buffered = union_all.buffer(buffer_distance)

    if buffered.geom_type == "Polygon":
        baseline_geom = LineString(buffered.exterior.coords)
    elif buffered.geom_type == "MultiPolygon":
        largest = max(buffered.geoms, key=lambda p: p.area)
        baseline_geom = LineString(largest.exterior.coords)
    else:
        raise ValueError("No se pudo generar línea base.")

    return gpd.GeoDataFrame({'id': ['baseline']}, geometry=[baseline_geom], crs=shorelines_gdf.crs)


def visualizar_shorelines(gdf):
    """Visualización profesional"""
    ids = sorted(gdf['id'].tolist())
    gdf_sorted = gdf.set_index('id').loc[ids].reset_index()

    cmap = cm.get_cmap('tab20', len(ids))
    color_list = [cmap(i) for i in range(len(ids))]

    fig, ax = plt.subplots(figsize=(14, 12))
    for idx, row in gdf_sorted.iterrows():
        row_gdf = gpd.GeoDataFrame(geometry=[row.geometry], crs=gdf_sorted.crs)
        row_gdf.plot(ax=ax, color=color_list[idx], linewidth=1)

    handles = [mpatches.Patch(color=color_list[i], label=ids[i]) for i in range(len(ids))]
    ax.legend(handles=handles, loc='lower left', fontsize=8, frameon=True, title="Meses-Años")

    ax.set_title("Líneas de costa (DSAS-ready)")
    ax.set_xlabel("Longitud (°)")
    ax.set_ylabel("Latitud (°)")

    axins = inset_axes(ax, width="35%", height="35%", loc='upper right', borderpad=2)
    for idx, row in gdf_sorted.iterrows():
        row_gdf = gpd.GeoDataFrame(geometry=[row.geometry], crs=gdf_sorted.crs)
        row_gdf.plot(ax=axins, color=color_list[idx], linewidth=1)

    axins.set_xlim(-106.52, -106.46)
    axins.set_ylim(23.32, 23.37)

    rect = plt.Rectangle((-106.52, 23.32), 0.05, 0.05, linewidth=1.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.show()

# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN DEL PIPELINE

# Leer el shapefile de recorte
clip_gdf = gpd.read_file(clip_shapefile).to_crs(epsg=32613)

# Procesar todos los años
gdfs_total = []
for year in years:
    print(f"\nProcesando año {year}...")
    input_base = os.path.join(base_dir, f'{year}shapes', 'input')
    output_base = os.path.join(base_dir, f'{year}shapes', 'output')
    os.makedirs(output_base, exist_ok=True)

    meses_id = {mes: f"{label}_{year}" for mes, label in meses_dict.items()}
    gdfs_clipped = procesar_anio(input_base, output_base, clip_gdf, meses_id)
    gdfs_total.extend(gdfs_clipped)

# Limpieza y exportación final
if gdfs_total:
    merged = limpiar_unir_geodfs(gdfs_total)
    merged_wgs84 = merged.to_crs(epsg=4326)

    # Export GeoJSON DSAS
    output_dsas_geojson = os.path.join(base_dir, 'shapes_total_dsas.geojson')
    merged_wgs84.to_file(output_dsas_geojson, driver='GeoJSON')
    print(f"\n✅ DSAS GeoJSON exportado en: {output_dsas_geojson}")

    # Línea base automática
    baseline_gdf = generar_linea_base(merged_wgs84, buffer_distance=-0.0002)
    baseline_path = os.path.join(base_dir, 'baseline_dsas.geojson')
    baseline_gdf.to_file(baseline_path, driver='GeoJSON')
    print(f"✅ Línea base DSAS exportada en: {baseline_path}")

    # Visualización
    visualizar_shorelines(merged_wgs84)
else:
    print("No se generaron resultados.")
