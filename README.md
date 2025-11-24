# reubicacion
INSTRUCCIONES 
- Python 3.8+ instalado
- Conexión a Internet activa
- Abir el cmd e ir a la carpeta 

1 Instalar Dependencias

pip install -r requirements.txt

2 Ejecutar el Algoritmo

### Comando

python ciudad_15min_reordenamiento.py --place "San Juan de Miraflores, Lima, Peru" 


3 Ver Resultados

Todos los archivos se guardan en la carpeta `outputs_reordenamiento/`

4 Archivos Principales

1. **`comparison_map.html`** ⭐ 
   - **Abre en tu navegador**
   - Mapa interactivo 
2. **`comparison_metrics.csv`**
   - Tabla con todas las métricas de cobertura
   - Compara estado inicial vs optimizado
3. **`evolution_analysis.png`**
   - Evolución del algoritmo durante las generaciones
   - Convergencia de intercambios
4. **`*.geojson`**
   - Archivos geoespaciales con ubicaciones de hogares y servicios
   - Útiles para análisis en QGIS o ArcGIS

Problemas Comunes
-Hay ocaciones muy raras cuando se descarga los datos del open street map se puede trabar por favor reiniciar el programa
-Hay ocaciones en donde el programa se pausa por favor pulsar una tecla , hast ahora donde  pasa eso es en medio del proceso del nsga2 
 
