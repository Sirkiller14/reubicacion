"""
Configuración Avanzada - Sistema de Planificación Urbana

Este archivo contiene parámetros avanzados para usuarios expertos que desean
personalizar el comportamiento del algoritmo de optimización.

USO:
----
1. Importa este archivo en ciudad_15min_reordenamiento.py
2. Modifica los valores según tus necesidades
3. Ejecuta el sistema normalmente

NOTA: Solo modifica estos valores si entiendes su impacto en el algoritmo.
"""

# =============================================================================
# CONFIGURACIÓN DE SERVICIOS
# =============================================================================

# Definición de categorías de servicios y sus etiquetas OSM
# Puedes agregar o modificar categorías aquí
SERVICIOS_PERSONALIZADOS = {
    "salud": [
        {"amenity": ["hospital", "clinic", "doctors", "dentist", "pharmacy"]},
        {"healthcare": True}  # Cualquier etiqueta healthcare
    ],
    
    "educacion": [
        {"amenity": ["school", "college", "university", "kindergarten"]},
        {"building": ["school", "university"]}
    ],
    
    "areas_verdes": [
        {"leisure": ["park", "garden", "playground", "recreation_ground"]},
        {"landuse": ["recreation_ground", "grass", "meadow"]}
    ],
    
    "trabajo": [
        {"amenity": ["office", "coworking"]},
        {"landuse": ["commercial", "industrial", "retail"]},
        {"shop": True},
        {"building": ["commercial", "industrial", "office", "retail"]}
    ],
    
    # Nueva categoría: Transporte
    "transporte": [
        {"amenity": ["bus_station", "taxi"]},
        {"public_transport": True},
        {"railway": ["station", "halt", "tram_stop"]}
    ],
    
    # Nueva categoría: Servicios Públicos
    "servicios_publicos": [
        {"amenity": ["police", "fire_station", "post_office", "townhall"]},
        {"office": ["government"]}
    ]
}

# =============================================================================
# PARÁMETROS DE OPTIMIZACIÓN
# =============================================================================

class ConfiguracionOptimizacion:
    """
    Parámetros del algoritmo NSGA-II
    """
    
    # Tamaño de población
    # Mayor = Más diversidad pero más lento
    # Rango recomendado: 60-200
    POBLACION_DEFAULT = 100
    POBLACION_RAPIDA = 60
    POBLACION_PROFUNDA = 150
    
    # Número de generaciones
    # Mayor = Mejor convergencia pero más lento
    # Rango recomendado: 40-200
    GENERACIONES_DEFAULT = 80
    GENERACIONES_RAPIDA = 40
    GENERACIONES_PROFUNDA = 150
    
    # Probabilidad de cruce (crossover)
    # Controla cuántos individuos se cruzan en cada generación
    # Rango: 0.7 - 0.95
    PROB_CRUCE = 0.9
    
    # Probabilidad de mutación
    # Controla cuánta variación aleatoria se introduce
    # Rango: 0.01 - 0.2
    PROB_MUTACION = 0.05
    
    # Elitismo
    # Número de mejores individuos que se preservan sin cambios
    # Rango: 1 - 10
    ELITE_SIZE = 2


class ConfiguracionProblema:
    """
    Parámetros del problema de optimización
    """
    
    # Proporción ideal de servicios respecto al total de ubicaciones
    # Ejemplo: 0.075 = 7.5% del total serán servicios
    # Rango: 0.05 - 0.15
    PROPORCION_IDEAL_SERVICIOS = 0.075
    
    # Factor de peso para el balance de servicios (alpha_balance)
    # Mayor valor = Mayor énfasis en mantener la proporción ideal
    # Rango: 0.05 - 0.3
    ALPHA_BALANCE_DEFAULT = 0.15
    ALPHA_BALANCE_ALTO = 0.25  # Más énfasis en balance
    ALPHA_BALANCE_BAJO = 0.10  # Más énfasis en cobertura
    
    # Peso para considerar otras categorías en la optimización
    # Controla cuánto importa no empeorar otras categorías
    # Rango: 0.0 - 0.5
    BETA_OTRAS_CATEGORIAS = 0.1
    
    # Tolerancia para la restricción de número de hogares
    # Qué tan estricta es la restricción de mantener N hogares
    # Rango: 0 - 5 (en número de hogares)
    TOLERANCIA_HOGARES = 0


class ConfiguracionRed:
    """
    Parámetros de la red peatonal
    """
    
    # Velocidad de caminata (km/h)
    VELOCIDAD_CAMINATA_LENTA = 3.5   # Personas mayores
    VELOCIDAD_CAMINATA_NORMAL = 4.5  # Adulto promedio
    VELOCIDAD_CAMINATA_RAPIDA = 5.5  # Persona joven/deportista
    
    # Velocidad de bicicleta (km/h) - para futuras extensiones
    VELOCIDAD_BICICLETA = 15.0
    
    # Factor de penalización por pendiente
    # Multiplica el tiempo de viaje en calles con pendiente
    # 1.0 = sin penalización, >1.0 = penalización
    FACTOR_PENDIENTE = 1.2
    
    # Tipo de red a descargar
    # Opciones: 'walk', 'bike', 'drive', 'all'
    TIPO_RED = 'walk'


class ConfiguracionMuestreo:
    """
    Parámetros de muestreo de datos
    """
    
    # Número máximo de hogares a considerar
    # Reducir este valor acelera el cálculo pero reduce precisión
    MAX_HOGARES_RAPIDO = 1000
    MAX_HOGARES_NORMAL = 2000
    MAX_HOGARES_COMPLETO = 3500
    
    # Número máximo de candidatos para nuevos servicios
    # Durante la optimización, se consideran hasta N ubicaciones candidatas
    MAX_CANDIDATOS = 400
    
    # Semilla para reproducibilidad
    # Usar la misma semilla garantiza resultados reproducibles
    SEMILLA_RANDOM = 42


class ConfiguracionVisualizacion:
    """
    Parámetros de visualización
    """
    
    # Colores para cada categoría de servicio (formato HTML)
    COLORES_SERVICIOS = {
        "health": "#e74c3c",          # Rojo
        "education": "#3498db",        # Azul
        "greens": "#2ecc71",           # Verde
        "work": "#9b59b6",             # Morado
        "transporte": "#f39c12",       # Naranja
        "servicios_publicos": "#34495e"  # Gris oscuro
    }
    
    # Tamaño de marcadores en el mapa
    RADIO_HOGAR = 2
    RADIO_SERVICIO_INICIAL = 4
    RADIO_SERVICIO_OPTIMIZADO = 6
    
    # Opacidad de elementos (0.0 - 1.0)
    OPACIDAD_HOGARES = 0.6
    OPACIDAD_SERVICIOS = 0.8


class ConfiguracionExportacion:
    """
    Parámetros de exportación de resultados
    """
    
    # Formato de archivos geoespaciales
    # Opciones: 'GeoJSON', 'ESRI Shapefile', 'GPKG'
    FORMATO_SALIDA = 'GeoJSON'
    
    # Guardar archivos intermedios (cada iteración)
    GUARDAR_INTERMEDIOS = False
    
    # Guardar frente de Pareto completo
    GUARDAR_PARETO_COMPLETO = True
    
    # Generar gráficos adicionales
    GENERAR_GRAFICOS = True


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def obtener_config_rapida():
    """Retorna configuración optimizada para velocidad"""
    return {
        'poblacion': ConfiguracionOptimizacion.POBLACION_RAPIDA,
        'generaciones': ConfiguracionOptimizacion.GENERACIONES_RAPIDA,
        'max_hogares': ConfiguracionMuestreo.MAX_HOGARES_RAPIDO,
        'alpha_balance': ConfiguracionProblema.ALPHA_BALANCE_DEFAULT
    }


def obtener_config_profunda():
    """Retorna configuración para análisis profundo"""
    return {
        'poblacion': ConfiguracionOptimizacion.POBLACION_PROFUNDA,
        'generaciones': ConfiguracionOptimizacion.GENERACIONES_PROFUNDA,
        'max_hogares': ConfiguracionMuestreo.MAX_HOGARES_COMPLETO,
        'alpha_balance': ConfiguracionProblema.ALPHA_BALANCE_ALTO
    }


def obtener_config_balanceada():
    """Retorna configuración balanceada (default)"""
    return {
        'poblacion': ConfiguracionOptimizacion.POBLACION_DEFAULT,
        'generaciones': ConfiguracionOptimizacion.GENERACIONES_DEFAULT,
        'max_hogares': ConfiguracionMuestreo.MAX_HOGARES_NORMAL,
        'alpha_balance': ConfiguracionProblema.ALPHA_BALANCE_DEFAULT
    }


# =============================================================================
# PERFILES PREDEFINIDOS
# =============================================================================

PERFILES = {
    'rapido': {
        'descripcion': 'Optimizado para velocidad (5-10 minutos)',
        'config': obtener_config_rapida()
    },
    
    'balanceado': {
        'descripcion': 'Balance entre velocidad y precisión (15-30 minutos)',
        'config': obtener_config_balanceada()
    },
    
    'profundo': {
        'descripcion': 'Análisis exhaustivo (45-90 minutos)',
        'config': obtener_config_profunda()
    },
    
    'tesis': {
        'descripcion': 'Configuración para investigación académica',
        'config': {
            'poblacion': 120,
            'generaciones': 150,
            'max_hogares': 3000,
            'alpha_balance': 0.15,
            'iteraciones': 5
        }
    }
}


# =============================================================================
# VALIDACIÓN DE PARÁMETROS
# =============================================================================

def validar_configuracion(config):
    """
    Valida que los parámetros estén en rangos aceptables
    
    Args:
        config (dict): Diccionario con parámetros de configuración
        
    Returns:
        tuple: (es_valido, lista_de_errores)
    """
    errores = []
    
    # Validar población
    if 'poblacion' in config:
        if not (20 <= config['poblacion'] <= 300):
            errores.append(f"Población fuera de rango: {config['poblacion']} (recomendado: 20-300)")
    
    # Validar generaciones
    if 'generaciones' in config:
        if not (10 <= config['generaciones'] <= 500):
            errores.append(f"Generaciones fuera de rango: {config['generaciones']} (recomendado: 10-500)")
    
    # Validar alpha_balance
    if 'alpha_balance' in config:
        if not (0.01 <= config['alpha_balance'] <= 1.0):
            errores.append(f"Alpha balance fuera de rango: {config['alpha_balance']} (recomendado: 0.01-1.0)")
    
    # Validar max_hogares
    if 'max_hogares' in config:
        if not (100 <= config['max_hogares'] <= 10000):
            errores.append(f"Max hogares fuera de rango: {config['max_hogares']} (recomendado: 100-10000)")
    
    return len(errores) == 0, errores


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=== CONFIGURACIÓN AVANZADA ===\n")
    
    print("Perfiles disponibles:")
    for nombre, perfil in PERFILES.items():
        print(f"\n{nombre.upper()}:")
        print(f"  Descripción: {perfil['descripcion']}")
        print(f"  Configuración:")
        for k, v in perfil['config'].items():
            print(f"    - {k}: {v}")
    
    print("\n\nPara usar un perfil en tu script:")
    print("  from configuracion_avanzada import PERFILES")
    print("  config = PERFILES['tesis']['config']")
    print("  # Usar config en tu optimización")
    
    print("\n\nValidación de configuración personalizada:")
    config_test = {
        'poblacion': 100,
        'generaciones': 80,
        'max_hogares': 2000,
        'alpha_balance': 0.15
    }
    es_valido, errores = validar_configuracion(config_test)
    if es_valido:
        print("  ✅ Configuración válida")
    else:
        print("  ❌ Errores encontrados:")
        for error in errores:
            print(f"    - {error}")
