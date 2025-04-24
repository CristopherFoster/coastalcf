# create_functions.py

import os

# Lista de nombres de funciones/clases que deseas crear
function_names = [
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
    "exportar_raster_y_shapefile"
]

# Ajusta esta ruta a tu carpeta coastalcf/coastalcf:
folder_path = r"C:\Users\c4cfo\OneDrive\CENTROGEO-cfoster\1_ENSOCE\coastalcf\coastalcf"

# Crea la carpeta si no existe
os.makedirs(folder_path, exist_ok=True)

for func_name in function_names:
    # Nombre del archivo .py
    file_name = f"{func_name}.py"
    file_path = os.path.join(folder_path, file_name)
    
    # Contenido a escribir
    # Manejo especial para la "clase" glcmOp
    if func_name == "glcmOp":
        content = f"""# {func_name}.py

class {func_name}:
    \"\"\"
    Clase o contenedor para el cálculo de texturas GLCM.
    \"\"\"
    def glcm(self, product, params):
        \"\"\"
        Cálculo de la textura GLCM.
        \"\"\"
        pass
"""
    else:
        # Reemplaza 'product' y el docstring según lo que necesites en cada función
        content = f"""# {func_name}.py

def {func_name}(product):
    \"\"\"
    Función: {func_name}
    Descripción: Aquí va la lógica interna de la función.
    \"\"\"
    pass
"""

    # Escribe el archivo en disco
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Archivos creados exitosamente.")
