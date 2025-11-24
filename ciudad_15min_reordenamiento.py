"""
Sistema de Planificación Urbana con Reordenamiento  
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import os
import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize
    from pymoo.core.sampling import Sampling
    from pymoo.core.repair import Repair
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.callback import Callback
    # Suprimir advertencia sobre módulos compilados
    try:
        from pymoo.config import Config
        Config.warnings['not_compiled'] = False
    except:
        pass
    PYMOO_OK = True
except Exception:
    PYMOO_OK = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Backend sin GUI para servidores
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

# -----------------------------
# CONFIGURACIÓN DE SERVICIOS
# -----------------------------

OSM_QUERIES = {
    "health": [{"amenity": ["hospital", "clinic", "doctors", "dentist", "pharmacy"]}],
    "education": [{"amenity": ["school", "college", "university", "kindergarten"]}],
    "greens": [{"leisure": ["park", "garden", "playground"]}, {"landuse": ["recreation_ground"]}],
    "work": [
        {"amenity": ["office", "coworking"]},
        {"landuse": ["commercial", "industrial"]},
        {"shop": True},
    ],
}

RESIDENTIAL_BUILDING_TAGS = {"building": ["residential", "apartments", "house", "detached", "terrace"]}

# -----------------------------
# UTILIDADES DE CARGA DE DATOS
# -----------------------------

def load_place_boundary(place: str) -> gpd.GeoDataFrame:
    """Carga el límite del área geográfica"""
    gdf = ox.geocode_to_gdf(place)
    if gdf.empty:
        raise ValueError(f"No se pudo geocodificar el lugar: {place}")
    return gdf.to_crs(4326)


def load_walking_graph(boundary: gpd.GeoDataFrame, speed_kmh: float = 4.5) -> nx.MultiDiGraph:
    """Carga la red peatonal del área"""
    poly = boundary.geometry.iloc[0]
    G = ox.graph_from_polygon(poly, network_type="walk", simplify=True)
    G = ox.distance.add_edge_lengths(G)
    speed_mps = (speed_kmh * 1000) / 3600
    for u, v, k, data in G.edges(keys=True, data=True):
        length = data.get("length", 0.0) or 0.0
        data["travel_time"] = length / max(speed_mps, 0.1)
    return G


def _download_pois(boundary: gpd.GeoDataFrame, osm_filters: List[dict]) -> gpd.GeoDataFrame:
    """Descarga puntos de interés desde OpenStreetMap"""
    poly = boundary.geometry.iloc[0]
    gdfs = []
    for f in osm_filters:
        try:
            g = ox.geometries_from_polygon(poly, f)
            if not g.empty:
                gdfs.append(g)
        except Exception:
            continue
    if not gdfs:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    g = pd.concat(gdfs, axis=0)
    g = g.reset_index(drop=True)
    g = g[g.geometry.notna()].to_crs(4326)
    g["geometry"] = g.geometry.centroid
    return g[["geometry"]].dropna().drop_duplicates()


def load_services(boundary: gpd.GeoDataFrame) -> Dict[str, gpd.GeoDataFrame]:
    """Carga todos los servicios por categoría"""
    services = {}
    for cat, filters in OSM_QUERIES.items():
        g = _download_pois(boundary, filters)
        g["category"] = cat
        g["type"] = "service"
        services[cat] = g
    return services


def load_residences(boundary: gpd.GeoDataFrame, max_points: int = None) -> gpd.GeoDataFrame:
    """Carga ubicaciones de hogares
    
    Args:
        boundary: Límite del área geográfica
        max_points: Número máximo de hogares a cargar. Si es None, carga todos los encontrados.
    """
    poly = boundary.geometry.iloc[0]
    try:
        b = ox.geometries_from_polygon(poly, RESIDENTIAL_BUILDING_TAGS)
        b = b[b.geometry.notna()].to_crs(4326)
        b["geometry"] = b.geometry.centroid
        homes = b[["geometry"]].dropna().drop_duplicates()
    except Exception:
        homes = gpd.GeoDataFrame(geometry=[], crs=4326)
    
    if homes.empty:
        # Fallback: muestrear puntos dentro del polígono
        bounds = poly.envelope
        minx, miny, maxx, maxy = bounds.bounds
        pts = []
        rng = np.random.default_rng(42)
        fallback_limit = max_points if max_points is not None else 3000
        for _ in range(30000):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            p = Point(x, y)
            if poly.contains(p):
                pts.append(p)
            if max_points is not None and len(pts) >= max_points:
                break
        homes = gpd.GeoDataFrame(geometry=pts, crs=4326)
    
    # Solo limitar si se especificó max_points
    if max_points is not None and len(homes) > max_points:
        homes = homes.sample(max_points, random_state=42).reset_index(drop=True)
    
    homes["category"] = "home"
    homes["type"] = "home"
    return homes


def nearest_node_series(G: nx.MultiDiGraph, gdf: gpd.GeoDataFrame) -> pd.Series:
    """Encuentra el nodo más cercano en la red para cada punto"""
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    nn = ox.distance.nearest_nodes(G, xs, ys)
    return pd.Series(nn, index=gdf.index)


# -----------------------------
# EVALUACIÓN DE ACCESIBILIDAD
# -----------------------------

def calculate_coverage(
    G: nx.MultiDiGraph,
    homes: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
    threshold_min: float = 15.0,
    home_nodes_precomputed: pd.Series = None,
    serv_nodes_precomputed: pd.Series = None,
) -> Tuple[float, np.ndarray]:
    """
    Calcula la cobertura de accesibilidad
    Retorna: (cobertura_porcentaje, array_booleano_de_alcanzabilidad)
    
    CORRECCIÓN: Calcula la distancia desde cada hogar hacia el servicio más cercano
    para evitar inconsistencias donde casas cercanas tienen tiempos muy diferentes.
    """
    if services.empty or homes.empty:
        return 0.0, np.zeros(len(homes), dtype=bool)
    
    # Usar nodos precalculados si están disponibles (más eficiente y consistente)
    if home_nodes_precomputed is not None and len(home_nodes_precomputed) == len(homes):
        home_nodes = home_nodes_precomputed
    else:
        home_nodes = nearest_node_series(G, homes)
    
    if serv_nodes_precomputed is not None and len(serv_nodes_precomputed) == len(services):
        serv_nodes = serv_nodes_precomputed
    else:
        serv_nodes = nearest_node_series(G, services)
    
    uniq_serv_nodes = list(set(serv_nodes.dropna().tolist()))
    
    if not uniq_serv_nodes:
        return 0.0, np.zeros(len(homes), dtype=bool)
    
    # CORRECCIÓN: Calcular distancia desde cada nodo de hogar hacia el servicio más cercano
    # Para evitar inconsistencias donde casas cercanas tienen tiempos muy diferentes,
    # calculamos desde cada hogar hacia TODOS los servicios y tomamos el mínimo
    reachable = np.zeros(len(homes), dtype=bool)
    
    # Calcular distancias desde todos los servicios hacia todos los nodos alcanzables
    # Esto nos da: para cada nodo alcanzable, la distancia mínima a cualquier servicio
    try:
        lengths_from_services = nx.multi_source_dijkstra_path_length(G, uniq_serv_nodes, weight="travel_time")
    except Exception as e:
        # Si falla multi_source, usar enfoque alternativo
        print(f"[ADVERTENCIA] Error en multi_source_dijkstra: {e}")
        lengths_from_services = {}
        for serv_node in uniq_serv_nodes:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, serv_node, weight="travel_time")
                # Actualizar con mínimos
                for node, dist in lengths.items():
                    if node not in lengths_from_services or dist < lengths_from_services[node]:
                        lengths_from_services[node] = dist
            except Exception:
                continue
    
    # Para cada hogar, buscar la distancia mínima al servicio más cercano
    unique_home_nodes = {}
    for i, (idx, hn) in enumerate(home_nodes.items()):
        # Agrupar hogares por nodo para evitar cálculos duplicados
        if hn not in unique_home_nodes:
            unique_home_nodes[hn] = []
        unique_home_nodes[hn].append(i)
    
    # Calcular tiempo para cada nodo único de hogar
    for hn, indices in unique_home_nodes.items():
        if hn in lengths_from_services:
            # Tiempo en segundos desde este nodo al servicio más cercano
            t_seconds = lengths_from_services[hn]
            t_minutes = t_seconds / 60.0
        else:
            # Nodo no alcanzable: intentar calcular desde este nodo hacia los servicios
            t_minutes = np.inf
            for serv_node in uniq_serv_nodes:
                try:
                    if nx.has_path(G, hn, serv_node):
                        path_length = nx.shortest_path_length(G, hn, serv_node, weight="travel_time")
                        t_minutes_candidate = path_length / 60.0
                        if t_minutes_candidate < t_minutes:
                            t_minutes = t_minutes_candidate
                except Exception:
                    continue
        
        # Asignar el mismo tiempo a todos los hogares en este nodo
        for idx in indices:
            reachable[idx] = t_minutes <= threshold_min
    
    coverage = float(np.mean(reachable))
    return coverage, reachable


def evaluate_all_categories(
    G: nx.MultiDiGraph,
    homes: gpd.GeoDataFrame,
    services_by_cat: Dict[str, gpd.GeoDataFrame],
    minutes: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evalúa cobertura para todas las categorías"""
    metrics = {}
    reach_arrays = {}
    
    for cat, pois in services_by_cat.items():
        cov, reach = calculate_coverage(G, homes, pois, minutes)
        reach_arrays[cat] = reach
        metrics[f"cov_{cat}"] = cov
    
    # Cobertura integral: hogares que alcanzan TODAS las categorías
    reach_df = pd.DataFrame(reach_arrays, index=homes.index)
    reach_df.columns = [f"reach_{c}" for c in services_by_cat.keys()]
    reach_df["all_categories"] = reach_df.all(axis=1)
    metrics["cov_all"] = reach_df["all_categories"].mean()
    
    return reach_df, metrics


# -----------------------------
# PROBLEMA DE OPTIMIZACIÓN CON REORDENAMIENTO
# -----------------------------

class FeasibleSampling(Sampling):
    """Inicialización que garantiza soluciones factibles (exactamente n_homes hogares)"""
    
    def __init__(self, n_homes: int):
        super().__init__()
        self.n_homes = n_homes
    
    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        X = np.zeros((n_samples, n_var), dtype=int)
        
        # Usar semilla diferente para cada muestra para más diversidad
        rng = np.random.default_rng()
        for i in range(n_samples):
            # Crear solución factible: exactamente n_homes ceros (hogares)
            x = np.ones(n_var, dtype=int)
            # Seleccionar aleatoriamente n_homes posiciones para ser hogares (0)
            home_indices = rng.choice(n_var, size=self.n_homes, replace=False)
            x[home_indices] = 0
            X[i] = x
        
        return X


class FeasibleSamplingAllCategories(Sampling):
    """Inicialización para todas las categorías: 0=hogar, 1=health, 2=education, 3=greens, 4=work
    
    Genera soluciones cercanas a la configuración inicial con un porcentaje controlado de cambios.
    """
    
    def __init__(self, n_homes: int, n_health: int, n_education: int, n_greens: int, n_work: int,
                 initial_change_percentage: float = 0.02):
        """
        Args:
            initial_change_percentage: Porcentaje de cambios iniciales (0.05 = 5% por defecto)
        """
        super().__init__()
        self.n_homes = n_homes
        self.n_health = n_health
        self.n_education = n_education
        self.n_greens = n_greens
        self.n_work = n_work
        self.initial_change_percentage = initial_change_percentage
    
    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        X = np.zeros((n_samples, n_var), dtype=int)
        
        # Obtener configuración inicial del problema
        if hasattr(problem, 'initial_config'):
            initial_config = problem.initial_config.copy()
        else:
            # Fallback: no hay configuración inicial, usar comportamiento aleatorio
            initial_config = None
        
        rng = np.random.default_rng()
        total_assigned = self.n_homes + self.n_health + self.n_education + self.n_greens + self.n_work
        
        # Verificar que no excedamos el número de variables
        if total_assigned > n_var:
            raise ValueError(f"Total asignado ({total_assigned}) excede número de variables ({n_var})")
        
        for i in range(n_samples):
            if initial_config is not None:
                # ESTRATEGIA: Empezar con configuración inicial y hacer solo un porcentaje de cambios
                x = initial_config.copy()
                
                # Calcular número de cambios basado en porcentaje
                n_changes = int(n_var * self.initial_change_percentage)
                # Asegurar límites razonables: mínimo 1, máximo 50% del total
                n_changes = max(1, min(n_changes, n_var // 2))
                
                # Hacer n_changes intercambios aleatorios manteniendo las cantidades correctas
                for _ in range(n_changes):
                    # Seleccionar dos tipos diferentes al azar
                    types = [0, 1, 2, 3, 4]
                    type1, type2 = rng.choice(types, size=2, replace=False)
                    
                    # Encontrar ubicaciones de cada tipo
                    type1_indices = np.where(x == type1)[0]
                    type2_indices = np.where(x == type2)[0]
                    
                    if len(type1_indices) > 0 and len(type2_indices) > 0:
                        # Seleccionar un índice aleatorio de cada tipo
                        idx1 = rng.choice(type1_indices)
                        idx2 = rng.choice(type2_indices)
                        
                        # Intercambiar (esto mantiene las cantidades correctas)
                        x[idx1], x[idx2] = x[idx2], x[idx1]
            else:
                # Fallback: comportamiento original (aleatorio completo) si no hay initial_config
                x = np.zeros(n_var, dtype=int)
                indices = np.arange(n_var)
                rng.shuffle(indices)
                
                # Asignar hogares (0)
                if self.n_homes > 0:
                    x[indices[:self.n_homes]] = 0
                    start = self.n_homes
                else:
                    start = 0
                
                # Asignar health (1)
                if self.n_health > 0:
                    x[indices[start:start+self.n_health]] = 1
                    start += self.n_health
                
                # Asignar education (2)
                if self.n_education > 0:
                    x[indices[start:start+self.n_education]] = 2
                    start += self.n_education
                
                # Asignar greens (3)
                if self.n_greens > 0:
                    x[indices[start:start+self.n_greens]] = 3
                    start += self.n_greens
                
                # Asignar work (4)
                if self.n_work > 0:
                    x[indices[start:start+self.n_work]] = 4
            
            X[i] = x
        
        return X


class FeasibleRepair(Repair):
    """Reparador que asegura que las soluciones tengan exactamente n_homes hogares"""
    
    def __init__(self, n_homes: int):
        super().__init__()
        self.n_homes = n_homes
    
    def _do(self, problem, X, **kwargs):
        X_repaired = X.copy()
        # Usar semilla diferente para cada llamada para más diversidad
        rng = np.random.default_rng()
        
        for i, x in enumerate(X):
            n_homes_actual = int((x == 0).sum())
            
            if n_homes_actual != self.n_homes:
                # Reparar: ajustar el número de hogares
                if n_homes_actual < self.n_homes:
                    # Necesitamos más hogares: convertir algunos servicios en hogares
                    service_indices = np.where(x == 1)[0]
                    n_needed = self.n_homes - n_homes_actual
                    if len(service_indices) >= n_needed:
                        to_convert = rng.choice(service_indices, size=n_needed, replace=False)
                        x[to_convert] = 0
                else:
                    # Necesitamos menos hogares: convertir algunos hogares en servicios
                    home_indices = np.where(x == 0)[0]
                    n_to_remove = n_homes_actual - self.n_homes
                    if len(home_indices) >= n_to_remove:
                        to_convert = rng.choice(home_indices, size=n_to_remove, replace=False)
                        x[to_convert] = 1
                
                X_repaired[i] = x
        
        return X_repaired


class FeasibleRepairAllCategories(Repair):
    """Reparador para todas las categorías: mantiene números correctos de cada tipo"""
    
    def __init__(self, n_homes: int, n_health: int, n_education: int, n_greens: int, n_work: int):
        super().__init__()
        self.targets = {
            0: n_homes,
            1: n_health,
            2: n_education,
            3: n_greens,
            4: n_work
        }
    
    def _do(self, problem, X, **kwargs):
        X_repaired = X.copy()
        rng = np.random.default_rng()
        
        for i, x in enumerate(X):
            # Contar actuales
            actuals = {
                0: int((x == 0).sum()),
                1: int((x == 1).sum()),
                2: int((x == 2).sum()),
                3: int((x == 3).sum()),
                4: int((x == 4).sum())
            }
            
            # Reparar cada tipo
            for type_id in range(5):
                diff = actuals[type_id] - self.targets[type_id]
                
                if diff != 0:
                    if diff > 0:
                        # Demasiados de este tipo: convertir a otros tipos que faltan
                        type_indices = np.where(x == type_id)[0]
                        to_convert = rng.choice(type_indices, size=diff, replace=False)
                        
                        # Encontrar tipos que necesitan más
                        for other_type in range(5):
                            if other_type != type_id and actuals[other_type] < self.targets[other_type]:
                                needed = self.targets[other_type] - actuals[other_type]
                                convert_count = min(needed, len(to_convert))
                                if convert_count > 0:
                                    x[to_convert[:convert_count]] = other_type
                                    actuals[other_type] += convert_count
                                    actuals[type_id] -= convert_count
                                    to_convert = to_convert[convert_count:]
                                    if len(to_convert) == 0:
                                        break
                    else:
                        # Faltan de este tipo: convertir de otros tipos que sobran
                        needed = -diff
                        for other_type in range(5):
                            if other_type != type_id and actuals[other_type] > self.targets[other_type]:
                                available = actuals[other_type] - self.targets[other_type]
                                convert_count = min(needed, available)
                                if convert_count > 0:
                                    other_indices = np.where(x == other_type)[0]
                                    to_convert = rng.choice(other_indices, size=convert_count, replace=False)
                                    x[to_convert] = type_id
                                    actuals[type_id] += convert_count
                                    actuals[other_type] -= convert_count
                                    needed -= convert_count
                                    if needed == 0:
                                        break
            
            X_repaired[i] = x
        
        return X_repaired


class FeasibleCrossoverAllCategories(Crossover):
    """Crossover para variables categóricas que mantiene números correctos de cada tipo"""
    
    def __init__(self, n_homes: int, n_health: int, n_education: int, n_greens: int, n_work: int, prob=0.9):
        super().__init__(2, 2)
        self.targets = {
            0: n_homes,
            1: n_health,
            2: n_education,
            3: n_greens,
            4: n_work
        }
        self.prob = prob
    
    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        n_offsprings = 2  # 2 descendientes por pareja
        X_off = np.zeros((n_offsprings, n_matings, n_var), dtype=int)
        
        rng = np.random.default_rng()
        
        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            
            # Generar 2 descendientes
            for o in range(n_offsprings):
                if rng.random() < self.prob:
                    # Crossover: intercambiar tipos entre padres manteniendo números correctos
                    # Estrategia: identificar ubicaciones donde los padres difieren y hacer intercambios
                    diff_mask = (p1 != p2)
                    diff_indices = np.where(diff_mask)[0]
                    
                    if len(diff_indices) > 0:
                        # Intercambiar algunos de los índices donde difieren
                        n_swaps = max(1, int(len(diff_indices) * 0.3))  # Intercambiar 30% de las diferencias
                        swap_indices = rng.choice(diff_indices, size=min(n_swaps, len(diff_indices)), replace=False)
                        
                        if o == 0:
                            offspring = p1.copy()
                            offspring[swap_indices] = p2[swap_indices]
                        else:
                            offspring = p2.copy()
                            offspring[swap_indices] = p1[swap_indices]
                    else:
                        # Si no hay diferencias, alternar entre padres
                        offspring = p1.copy() if o == 0 else p2.copy()
                else:
                    # Sin crossover, alternar entre padres
                    offspring = p1.copy() if o == 0 else p2.copy()
                
                # Verificar y corregir si es necesario (pero solo si hay desbalance significativo)
                actuals = {
                    0: int((offspring == 0).sum()),
                    1: int((offspring == 1).sum()),
                    2: int((offspring == 2).sum()),
                    3: int((offspring == 3).sum()),
                    4: int((offspring == 4).sum())
                }
                
                # Solo reparar si hay desbalance grande
                total_diff = sum(abs(actuals[i] - self.targets[i]) for i in range(5))
                if total_diff > 5:  # Solo reparar si hay más de 5 diferencias
                    for type_id in range(5):
                        diff = actuals[type_id] - self.targets[type_id]
                        if diff > 0:
                            # Demasiados: convertir a otros que faltan
                            type_indices = np.where(offspring == type_id)[0]
                            to_convert = rng.choice(type_indices, size=diff, replace=False)
                            for other_type in range(5):
                                if other_type != type_id and actuals[other_type] < self.targets[other_type]:
                                    needed = self.targets[other_type] - actuals[other_type]
                                    convert_count = min(needed, len(to_convert))
                                    if convert_count > 0:
                                        offspring[to_convert[:convert_count]] = other_type
                                        actuals[other_type] += convert_count
                                        actuals[type_id] -= convert_count
                                        to_convert = to_convert[convert_count:]
                                        if len(to_convert) == 0:
                                            break
                        elif diff < 0:
                            # Faltan: convertir de otros que sobran
                            needed = -diff
                            for other_type in range(5):
                                if other_type != type_id and actuals[other_type] > self.targets[other_type]:
                                    available = actuals[other_type] - self.targets[other_type]
                                    convert_count = min(needed, available)
                                    if convert_count > 0:
                                        other_indices = np.where(offspring == other_type)[0]
                                        to_convert = rng.choice(other_indices, size=convert_count, replace=False)
                                        offspring[to_convert] = type_id
                                        actuals[type_id] += convert_count
                                        actuals[other_type] -= convert_count
                                        needed -= convert_count
                                        if needed == 0:
                                            break
                
                X_off[o, k] = offspring
        
        return X_off


class FeasibleMutationAllCategories(Mutation):
    """Mutación para variables categóricas que intercambia tipos manteniendo números correctos"""
    
    def __init__(self, n_homes: int, n_health: int, n_education: int, n_greens: int, n_work: int, prob=0.7):
        super().__init__()
        self.targets = {
            0: n_homes,
            1: n_health,
            2: n_education,
            3: n_greens,
            4: n_work
        }
        self.prob = prob
    
    def _do(self, problem, X, **kwargs):
        X_mut = X.copy()
        rng = np.random.default_rng()
        
        for i in range(len(X)):
            if rng.random() < self.prob:
                x = X[i].copy()
                
                # Estrategia: intercambiar tipos entre ubicaciones para mantener números correctos
                # Seleccionar dos tipos diferentes al azar
                types = [0, 1, 2, 3, 4]
                type1, type2 = rng.choice(types, size=2, replace=False)
                
                # Encontrar ubicaciones de cada tipo
                type1_indices = np.where(x == type1)[0]
                type2_indices = np.where(x == type2)[0]
                
                # Intercambiar algunas ubicaciones (máximo 5% de cada tipo)
                n_swaps = max(1, int(min(len(type1_indices), len(type2_indices)) * 0.05))
                n_swaps = min(n_swaps, len(type1_indices), len(type2_indices))
                
                if n_swaps > 0:
                    swap1 = rng.choice(type1_indices, size=n_swaps, replace=False)
                    swap2 = rng.choice(type2_indices, size=n_swaps, replace=False)
                    
                    # Intercambiar
                    x[swap1] = type2
                    x[swap2] = type1
                
                X_mut[i] = x
        
        return X_mut


class ReorderingProblem(ElementwiseProblem):
    """
    Problema de optimización que permite intercambiar posiciones entre hogares y servicios.
    
    ENFOQUE:
    - Mantiene constante el número de hogares
    - Las variables representan ASIGNACIONES de ubicaciones a tipos (hogar o servicio)
    - Cada ubicación puede ser: hogar, servicio_salud, servicio_educación, etc.
    """
    
    def __init__(self, 
                 G: nx.MultiDiGraph,
                 initial_homes: gpd.GeoDataFrame,
                 initial_services: Dict[str, gpd.GeoDataFrame],
                 target_category: str,
                 minutes: float = 15.0,
                 alpha_balance: float = 0.1):
        """
        Args:
            G: Grafo de la red peatonal
            initial_homes: Ubicaciones iniciales de hogares
            initial_services: Servicios iniciales por categoría
            target_category: Categoría de servicio a optimizar
            minutes: Umbral de minutos para accesibilidad
            alpha_balance: Factor de peso para el balance de servicios
        """
        self.G = G
        self.initial_homes = initial_homes.copy()
        self.initial_services = {k: v.copy() for k, v in initial_services.items()}
        self.target_category = target_category
        self.minutes = minutes
        self.alpha_balance = alpha_balance
        
        # Número fijo de hogares (debe mantenerse constante)
        self.n_homes = len(initial_homes)
        
        # Crear pool de ubicaciones: todos los puntos disponibles
        all_locations = [initial_homes]
        for cat_services in initial_services.values():
            if not cat_services.empty:
                all_locations.append(cat_services)
        
        self.location_pool = pd.concat(all_locations, ignore_index=True)
        self.location_pool = self.location_pool[['geometry']].drop_duplicates().reset_index(drop=True)
        
        n_locations = len(self.location_pool)
        
        # Variables: para cada ubicación, asignar un tipo
        # 0 = hogar, 1 = servicio de la categoría target
        # Restricción: exactamente n_homes deben ser hogares
        super().__init__(
            n_var=n_locations,
            n_obj=2,
            n_constr=1,
            xl=0,
            xu=1,
            type_var=np.int64
        )
        
        # Pre-computar nodos más cercanos
        self.location_nodes = nearest_node_series(G, self.location_pool)
        
        print(f"[Problema Inicializado]")
        print(f"  - Ubicaciones totales: {n_locations}")
        print(f"  - Hogares a mantener: {self.n_homes}")
        print(f"  - Categoría objetivo: {target_category}")
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evalúa una solución:
        x: array donde x[i] = 0 (hogar) o 1 (servicio)
        """
        # Separar hogares y servicios según la asignación
        home_mask = (x == 0)
        service_mask = (x == 1)
        
        homes_locs = self.location_pool[home_mask].copy()
        service_locs = self.location_pool[service_mask].copy()
        
        # Calcular cobertura para la categoría objetivo
        if not service_locs.empty and not homes_locs.empty:
            cov_target, _ = calculate_coverage(
                self.G, homes_locs, service_locs, self.minutes
            )
        else:
            cov_target = 0.0
        
        # Calcular cobertura para otras categorías (mantener servicios existentes)
        other_coverage = []
        for cat, serv_gdf in self.initial_services.items():
            if cat != self.target_category and not serv_gdf.empty and not homes_locs.empty:
                cov, _ = calculate_coverage(self.G, homes_locs, serv_gdf, self.minutes)
                other_coverage.append(cov)
        
        avg_other_cov = np.mean(other_coverage) if other_coverage else 0.0
        
        # Objetivos:
        # f1: Minimizar (1 - cobertura_objetivo) -> maximizar cobertura
        # f2: Balance - penalizar si hay demasiados o muy pocos servicios
        n_services = int(service_mask.sum())
        n_homes = int(home_mask.sum())
        
        # Proporción ideal de servicios: ~5-10% del total
        ideal_service_ratio = 0.075
        service_ratio = n_services / len(x)
        balance_penalty = abs(service_ratio - ideal_service_ratio) / ideal_service_ratio
        
        f1 = 1.0 - cov_target
        f2 = self.alpha_balance * balance_penalty + 0.1 * (1.0 - avg_other_cov)
        
        # Restricción: debe haber exactamente n_homes hogares (con pequeño margen)
        # Permitimos un margen del 1% para facilitar la convergencia
        margin = max(1, int(self.n_homes * 0.01))
        g1 = max(0, abs(n_homes - self.n_homes) - margin)
        
        out["F"] = [f1, f2]
        out["G"] = [g1]


class ReorderingProblemAllCategories(ElementwiseProblem):
    """
    Problema de optimización que optimiza TODAS las categorías simultáneamente.
    
    ENFOQUE:
    - Variables categóricas: 0=hogar, 1=health, 2=education, 3=greens, 4=work
    - Optimiza cobertura de todas las categorías al mismo tiempo
    - Mantiene números fijos de cada tipo
    """
    
    def __init__(self, 
                 G: nx.MultiDiGraph,
                 initial_homes: gpd.GeoDataFrame,
                 initial_services: Dict[str, gpd.GeoDataFrame],
                 minutes: float = 15.0):
        """
        Args:
            G: Grafo de la red peatonal
            initial_homes: Ubicaciones iniciales de hogares
            initial_services: Servicios iniciales por categoría
            minutes: Umbral de minutos para accesibilidad
        """
        self.G = G
        self.initial_homes = initial_homes.copy()
        self.initial_services = {k: v.copy() for k, v in initial_services.items()}
        self.minutes = minutes
        
        # Números objetivo iniciales de cada tipo
        n_homes_initial = len(initial_homes)
        n_health_initial = len(initial_services.get("health", gpd.GeoDataFrame()))
        n_education_initial = len(initial_services.get("education", gpd.GeoDataFrame()))
        n_greens_initial = len(initial_services.get("greens", gpd.GeoDataFrame()))
        n_work_initial = len(initial_services.get("work", gpd.GeoDataFrame()))
        
        # Crear pool de ubicaciones: todos los puntos disponibles
        # Primero marcar cada ubicación con su tipo inicial
        initial_homes_marked = initial_homes.copy()
        initial_homes_marked['initial_type'] = 0  # 0 = hogar
        
        all_locations = [initial_homes_marked]
        category_map_init = {"health": 1, "education": 2, "greens": 3, "work": 4}
        for cat, cat_services in initial_services.items():
            if not cat_services.empty:
                cat_marked = cat_services.copy()
                cat_marked['initial_type'] = category_map_init.get(cat, 1)  # 1=health, 2=education, 3=greens, 4=work
                all_locations.append(cat_marked)
        
        self.location_pool = pd.concat(all_locations, ignore_index=True)
        
        # Para duplicados, conservar el primer tipo encontrado (preferencia: hogares primero)
        self.location_pool = self.location_pool.sort_values('initial_type').drop_duplicates(subset=['geometry'], keep='first').reset_index(drop=True)
        
        # Guardar configuración inicial
        self.initial_config = self.location_pool['initial_type'].values
        
        # Eliminar columna 'initial_type' para mantener solo geometry
        self.location_pool = self.location_pool[['geometry']].reset_index(drop=True)
        
        n_locations = len(self.location_pool)
        
        # Ajustar números objetivo proporcionalmente si hay duplicados eliminados
        total_initial = n_homes_initial + n_health_initial + n_education_initial + n_greens_initial + n_work_initial
        
        if total_initial > n_locations:
            # Hay duplicados, ajustar proporcionalmente
            ratio = n_locations / total_initial
            self.n_homes = max(1, int(n_homes_initial * ratio))
            self.n_health = max(0, int(n_health_initial * ratio))
            self.n_education = max(0, int(n_education_initial * ratio))
            self.n_greens = max(0, int(n_greens_initial * ratio))
            self.n_work = max(0, int(n_work_initial * ratio))
            
            # Ajustar para que la suma sea exactamente n_locations
            current_sum = self.n_homes + self.n_health + self.n_education + self.n_greens + self.n_work
            diff = n_locations - current_sum
            
            if diff != 0:
                # Ajustar principalmente los hogares para mantener la proporción
                self.n_homes += diff
                if self.n_homes < 1:
                    self.n_homes = 1
                    # Ajustar otros tipos si es necesario
                    remaining = n_locations - self.n_homes - self.n_health - self.n_education - self.n_greens - self.n_work
                    if remaining > 0:
                        self.n_health += remaining
                    elif remaining < 0:
                        self.n_health = max(0, self.n_health + remaining)
        else:
            # No hay duplicados, usar números originales
            self.n_homes = n_homes_initial
            self.n_health = n_health_initial
            self.n_education = n_education_initial
            self.n_greens = n_greens_initial
            self.n_work = n_work_initial
        
        # Variables categóricas: 0=hogar, 1=health, 2=education, 3=greens, 4=work
        # 5 objetivos: 4 de cobertura + 1 de minimización de cambios
        # 5 restricciones (una por cada tipo)
        super().__init__(
            n_var=n_locations,
            n_obj=5,  # health, education, greens, work, y minimizar cambios
            n_constr=5,  # Restricciones para cada tipo
            xl=0,
            xu=4,  # 0-4 para los 5 tipos
            type_var=np.int64
        )
        
        # Pre-computar nodos más cercanos
        self.location_nodes = nearest_node_series(G, self.location_pool)
        
        # Mapeo de categorías a números
        self.category_map = {"health": 1, "education": 2, "greens": 3, "work": 4}
        
        print(f"[Problema Inicializado - Todas las Categorías]")
        print(f"  - Ubicaciones totales: {n_locations}")
        print(f"  - Total inicial (antes de eliminar duplicados): {total_initial}")
        if total_initial > n_locations:
            print(f"  - Duplicados eliminados: {total_initial - n_locations}")
        print(f"  - Hogares: {self.n_homes}")
        print(f"  - Health: {self.n_health}")
        print(f"  - Education: {self.n_education}")
        print(f"  - Greens: {self.n_greens}")
        print(f"  - Work: {self.n_work}")
        print(f"  - Total asignado: {self.n_homes + self.n_health + self.n_education + self.n_greens + self.n_work}")
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evalúa una solución:
        x: array donde x[i] = 0 (hogar), 1 (health), 2 (education), 3 (greens), 4 (work)
        """
        # Separar por tipo
        homes_locs = self.location_pool[x == 0].copy()
        health_locs = self.location_pool[x == 1].copy()
        education_locs = self.location_pool[x == 2].copy()
        greens_locs = self.location_pool[x == 3].copy()
        work_locs = self.location_pool[x == 4].copy()
        
        # Calcular cobertura para cada categoría
        objectives = []
        
        # Usar nodos precalculados para consistencia
        home_mask_indices = np.where(x == 0)[0]
        health_mask_indices = np.where(x == 1)[0]
        education_mask_indices = np.where(x == 2)[0]
        greens_mask_indices = np.where(x == 3)[0]
        work_mask_indices = np.where(x == 4)[0]
        
        # f1: Minimizar (1 - cobertura_health)
        if not health_locs.empty and not homes_locs.empty:
            home_nodes_subset = self.location_nodes.iloc[home_mask_indices] if len(home_mask_indices) > 0 else None
            serv_nodes_subset = self.location_nodes.iloc[health_mask_indices] if len(health_mask_indices) > 0 else None
            cov_health, _ = calculate_coverage(
                self.G, homes_locs, health_locs, self.minutes,
                home_nodes_precomputed=home_nodes_subset,
                serv_nodes_precomputed=serv_nodes_subset
            )
            objectives.append(1.0 - cov_health)
        else:
            objectives.append(1.0)
        
        # f2: Minimizar (1 - cobertura_education)
        if not education_locs.empty and not homes_locs.empty:
            home_nodes_subset = self.location_nodes.iloc[home_mask_indices] if len(home_mask_indices) > 0 else None
            serv_nodes_subset = self.location_nodes.iloc[education_mask_indices] if len(education_mask_indices) > 0 else None
            cov_education, _ = calculate_coverage(
                self.G, homes_locs, education_locs, self.minutes,
                home_nodes_precomputed=home_nodes_subset,
                serv_nodes_precomputed=serv_nodes_subset
            )
            objectives.append(1.0 - cov_education)
        else:
            objectives.append(1.0)
        
        # f3: Minimizar (1 - cobertura_greens)
        if not greens_locs.empty and not homes_locs.empty:
            home_nodes_subset = self.location_nodes.iloc[home_mask_indices] if len(home_mask_indices) > 0 else None
            serv_nodes_subset = self.location_nodes.iloc[greens_mask_indices] if len(greens_mask_indices) > 0 else None
            cov_greens, _ = calculate_coverage(
                self.G, homes_locs, greens_locs, self.minutes,
                home_nodes_precomputed=home_nodes_subset,
                serv_nodes_precomputed=serv_nodes_subset
            )
            objectives.append(1.0 - cov_greens)
        else:
            objectives.append(1.0)
        
        # f4: Minimizar (1 - cobertura_work)
        if not work_locs.empty and not homes_locs.empty:
            home_nodes_subset = self.location_nodes.iloc[home_mask_indices] if len(home_mask_indices) > 0 else None
            serv_nodes_subset = self.location_nodes.iloc[work_mask_indices] if len(work_mask_indices) > 0 else None
            cov_work, _ = calculate_coverage(
                self.G, homes_locs, work_locs, self.minutes,
                home_nodes_precomputed=home_nodes_subset,
                serv_nodes_precomputed=serv_nodes_subset
            )
            objectives.append(1.0 - cov_work)
        else:
            objectives.append(1.0)
        
        # f5: Minimizar número de cambios respecto a la configuración inicial
        # Contar cuántas ubicaciones cambiaron de tipo
        n_changes = int((x != self.initial_config).sum())
        # Normalizar por número total de ubicaciones (para que esté entre 0 y 1)
        change_ratio = n_changes / len(x) if len(x) > 0 else 1.0
        # PESO MODERADO: Multiplicar por 5.0 para penalizar los cambios durante la evolución
        # Esto hace que el algoritmo prefiera soluciones con menos cambios, pero permite
        # más exploración que con pesos extremadamente altos. Con 5.0, los cambios tienen
        # un costo moderado, balanceando entre mantener la configuración inicial y mejorar cobertura.
        objectives.append(change_ratio * 5.0)
        
        # Restricciones: debe haber exactamente el número objetivo de cada tipo
        margin = max(1, int(min(self.n_homes, self.n_health, self.n_education, self.n_greens, self.n_work) * 0.01))
        
        n_homes_actual = int((x == 0).sum())
        n_health_actual = int((x == 1).sum())
        n_education_actual = int((x == 2).sum())
        n_greens_actual = int((x == 3).sum())
        n_work_actual = int((x == 4).sum())
        
        g1 = max(0, abs(n_homes_actual - self.n_homes) - margin)
        g2 = max(0, abs(n_health_actual - self.n_health) - margin)
        g3 = max(0, abs(n_education_actual - self.n_education) - margin)
        g4 = max(0, abs(n_greens_actual - self.n_greens) - margin)
        g5 = max(0, abs(n_work_actual - self.n_work) - margin)
        
        out["F"] = objectives
        out["G"] = [g1, g2, g3, g4, g5]


# -----------------------------
# TRACKING DE INTERCAMBIOS Y CALLBACK PERSONALIZADO
# -----------------------------

@dataclass
class ExchangeTracker:
    """Clase para rastrear intercambios en individuos"""
    generation: int
    individual_index: int
    n_exchanges: int  # Número de intercambios (cambios respecto a inicial)
    objectives: List[float] = field(default_factory=list)
    solution: Optional[np.ndarray] = None


class EvolutionCallback(Callback):
    """
    Callback personalizado para NSGA-II que rastrea intercambios por generación.
    Captura datos en tiempo real sin afectar el rendimiento.
    """
    
    def __init__(self, initial_config: np.ndarray, track_generations: List[int] = None):
        """
        Args:
            initial_config: Configuración inicial (referencia para calcular intercambios)
            track_generations: Lista de generaciones específicas a capturar (ej: [1, 2, 3, 78, 79, 80])
        """
        super().__init__()
        self.initial_config = initial_config.copy()
        self.track_generations = track_generations if track_generations else list(range(1, 11)) + list(range(95, 105)) + list(range(191, 201))
        
        # Almacenamiento de datos
        self.generation_data: Dict[int, Dict] = {}
        self.exchange_tracking: List[ExchangeTracker] = []
        self.evolution_history: List[Dict] = []
    
    def notify(self, algorithm):
        """Se llama en cada generación"""
        try:
            generation = algorithm.n_gen
            
            # Calcular estadísticas de la población actual
            # En pymoo 0.6.1.1, el algoritmo tiene una población con X y F
            pop = algorithm.pop
            if pop is None:
                return
            
            # Intentar diferentes formas de acceder a los datos
            X = None
            F = None
            
            if hasattr(pop, 'get'):
                X = pop.get("X")
                F = pop.get("F")
            elif hasattr(pop, 'X') and hasattr(pop, 'F'):
                X = pop.X
                F = pop.F
            elif hasattr(pop, '__getitem__'):
                try:
                    X = pop["X"]
                    F = pop["F"]
                except:
                    pass
            
            if X is None or F is None or len(X) == 0:
                return
        except Exception as e:
            # Si hay error al acceder a los datos, ignorar esta generación
            # (no afecta el rendimiento de la optimización)
            return
        
        # Calcular intercambios para cada individuo
        exchanges_per_individual = []
        for i, x in enumerate(X):
            # Número de intercambios = posiciones donde cambió respecto a inicial
            n_exchanges = int((x != self.initial_config).sum())
            exchanges_per_individual.append(n_exchanges)
            
            # Capturar individuos de generaciones específicas
            if generation in self.track_generations:
                tracker = ExchangeTracker(
                    generation=generation,
                    individual_index=i,
                    n_exchanges=n_exchanges,
                    objectives=F[i].tolist() if len(F) > i else [],
                    solution=x.copy()
                )
                self.exchange_tracking.append(tracker)
        
        # Estadísticas por generación
        stats = {
            "generation": generation,
            "mean_exchanges": float(np.mean(exchanges_per_individual)),
            "std_exchanges": float(np.std(exchanges_per_individual)),
            "min_exchanges": int(np.min(exchanges_per_individual)),
            "max_exchanges": int(np.max(exchanges_per_individual)),
            "median_exchanges": float(np.median(exchanges_per_individual)),
            "n_individuals": len(X),
            "best_objective": float(np.min(F[:, 0])) if len(F) > 0 and F.shape[1] > 0 else 0.0,
            "mean_objective": float(np.mean(F[:, 0])) if len(F) > 0 and F.shape[1] > 0 else 0.0
        }
        
        self.generation_data[generation] = stats
        self.evolution_history.append(stats)
    
    def get_exchange_stats(self) -> pd.DataFrame:
        """Obtiene estadísticas de intercambios como DataFrame"""
        if not self.evolution_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.evolution_history)
    
    def get_tracked_exchanges(self) -> pd.DataFrame:
        """Obtiene intercambios capturados de generaciones específicas"""
        if not self.exchange_tracking:
            return pd.DataFrame()
        
        data = []
        for tracker in self.exchange_tracking:
            data.append({
                "generation": tracker.generation,
                "individual_index": tracker.individual_index,
                "n_exchanges": tracker.n_exchanges,
                "objectives": tracker.objectives,
            })
        
        return pd.DataFrame(data)
    
    def export_detailed_stats(self, output_dir: str):
        """Exporta estadísticas detalladas a CSV y JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Estadísticas por generación
        stats_df = self.get_exchange_stats()
        if not stats_df.empty:
            stats_df.to_csv(
                os.path.join(output_dir, "exchange_stats_by_generation.csv"),
                index=False
            )
        
        # Intercambios capturados de generaciones específicas
        tracked_df = self.get_tracked_exchanges()
        if not tracked_df.empty:
            tracked_df.to_csv(
                os.path.join(output_dir, "tracked_exchanges_specific_generations.csv"),
                index=False
            )
        
        # Estadísticas agregadas en JSON
        summary = {
            "total_generations_tracked": len(self.generation_data),
            "tracked_generations": sorted(self.track_generations),
            "total_individuals_captured": len(self.exchange_tracking),
            "statistics_by_generation": {
                str(gen): stats for gen, stats in self.generation_data.items()
            }
        }
        
        with open(os.path.join(output_dir, "evolution_summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def calculate_exchanges(initial_config: np.ndarray, solution: np.ndarray) -> int:
    """
    Calcula el número de intercambios (cambios) entre configuración inicial y solución
    
    Args:
        initial_config: Configuración inicial
        solution: Solución actual
    
    Returns:
        Número de intercambios (posiciones que cambiaron)
    """
    return int((solution != initial_config).sum())


# -----------------------------
# ANÁLISIS EVOLUTIVO Y VISUALIZACIÓN
# -----------------------------

def plot_exchange_evolution(callback: EvolutionCallback, output_dir: str):
    """
    Genera gráficos de evolución de intercambios
    
    Args:
        callback: Callback con datos de evolución
        output_dir: Directorio donde guardar los gráficos
    """
    if not MATPLOTLIB_OK:
        print("[ADVERTENCIA] matplotlib no instalado: omitiendo gráficos")
        return
    
    if not callback.evolution_history:
        print("[ADVERTENCIA] No hay datos de evolución para graficar")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    stats_df = callback.get_exchange_stats()
    
    if stats_df.empty:
        return
    
    # 1. Evolución de intercambios promedio por generación
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis Evolutivo de Intercambios', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Evolución de intercambios promedio
    ax1 = axes[0, 0]
    ax1.plot(stats_df['generation'], stats_df['mean_exchanges'], 
             label='Promedio', linewidth=2, color='blue')
    ax1.fill_between(stats_df['generation'], 
                     stats_df['mean_exchanges'] - stats_df['std_exchanges'],
                     stats_df['mean_exchanges'] + stats_df['std_exchanges'],
                     alpha=0.3, color='blue', label='±1 Desviación Estándar')
    ax1.plot(stats_df['generation'], stats_df['min_exchanges'], 
             '--', label='Mínimo', linewidth=1, color='green', alpha=0.7)
    ax1.plot(stats_df['generation'], stats_df['max_exchanges'], 
             '--', label='Máximo', linewidth=1, color='red', alpha=0.7)
    ax1.set_xlabel('Generación', fontsize=11)
    ax1.set_ylabel('Número de Intercambios', fontsize=11)
    ax1.set_title('Evolución de Intercambios por Generación', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Distribución de intercambios (boxplot por generaciones seleccionadas)
    ax2 = axes[0, 1]
    tracked_df = callback.get_tracked_exchanges()
    if not tracked_df.empty:
        tracked_generations = sorted(tracked_df['generation'].unique())
        box_data = [tracked_df[tracked_df['generation'] == gen]['n_exchanges'].values 
                   for gen in tracked_generations]
        bp = ax2.boxplot(box_data, labels=[f'Gen {g}' for g in tracked_generations], 
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_xlabel('Generación', fontsize=11)
        ax2.set_ylabel('Número de Intercambios', fontsize=11)
        ax2.set_title('Distribución de Intercambios en Generaciones Específicas', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'No hay datos para mostrar', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Distribución de Intercambios', fontsize=12)
    
    # Gráfico 3: Convergencia del algoritmo (objetivo)
    ax3 = axes[1, 0]
    if 'best_objective' in stats_df.columns and 'mean_objective' in stats_df.columns:
        ax3.plot(stats_df['generation'], stats_df['best_objective'], 
                label='Mejor Objetivo', linewidth=2, color='green')
        ax3.plot(stats_df['generation'], stats_df['mean_objective'], 
                label='Objetivo Promedio', linewidth=2, color='orange')
        ax3.set_xlabel('Generación', fontsize=11)
        ax3.set_ylabel('Valor del Objetivo', fontsize=11)
        ax3.set_title('Convergencia del Algoritmo', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Datos de objetivos no disponibles', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Convergencia del Algoritmo', fontsize=12)
    
    # Gráfico 4: Histograma de intercambios en generaciones específicas
    ax4 = axes[1, 1]
    if not tracked_df.empty:
        all_exchanges = tracked_df['n_exchanges'].values
        ax4.hist(all_exchanges, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax4.axvline(np.mean(all_exchanges), color='red', linestyle='--', 
                   linewidth=2, label=f'Promedio: {np.mean(all_exchanges):.1f}')
        ax4.axvline(np.median(all_exchanges), color='green', linestyle='--', 
                   linewidth=2, label=f'Mediana: {np.median(all_exchanges):.1f}')
        ax4.set_xlabel('Número de Intercambios', fontsize=11)
        ax4.set_ylabel('Frecuencia', fontsize=11)
        ax4.set_title('Distribución de Intercambios (Todas las Generaciones Capturadas)', 
                     fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No hay datos para mostrar', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Distribución de Intercambios', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "evolution_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Gráfico de evolución guardado en: {output_path}")
    
    # Gráfico adicional: Comparación entre generaciones específicas
    if not tracked_df.empty:
        fig2, ax = plt.subplots(figsize=(12, 6))
        tracked_generations = sorted(tracked_df['generation'].unique())
        
        positions = np.arange(len(tracked_generations))
        means = [tracked_df[tracked_df['generation'] == gen]['n_exchanges'].mean() 
                for gen in tracked_generations]
        stds = [tracked_df[tracked_df['generation'] == gen]['n_exchanges'].std() 
               for gen in tracked_generations]
        
        bars = ax.bar(positions, means, yerr=stds, capsize=5, alpha=0.7, 
                     color='steelblue', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Generación', fontsize=12, fontweight='bold')
        ax.set_ylabel('Intercambios Promedio', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Intercambios en Generaciones Específicas', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels([f'Gen {g}' for g in tracked_generations])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores en las barras
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.5,
                   f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path2 = os.path.join(output_dir, "exchange_comparison_generations.png")
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Gráfico de comparación guardado en: {output_path2}")


def plot_distribution_by_periods(callback: EvolutionCallback, output_dir: str, max_gen: int = 200):
    """
    Genera 3 gráficos de distribución de intercambios:
    1. Primeras 10 generaciones
    2. 10 generaciones del medio
    3. Últimas 10 generaciones
    
    Args:
        callback: Callback con datos de evolución
        output_dir: Directorio donde guardar los gráficos
        max_gen: Número máximo de generaciones
    """
    if not MATPLOTLIB_OK:
        print("[ADVERTENCIA] matplotlib no instalado: omitiendo gráficos")
        return
    
    tracked_df = callback.get_tracked_exchanges()
    if tracked_df.empty:
        print("[ADVERTENCIA] No hay datos de intercambios para graficar")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Definir los 3 rangos de generaciones
    first_gen = list(range(1, 11))  # Primeras 10: 1-10
    mid_start = max(1, (max_gen // 2) - 4)
    mid_end = min(max_gen, mid_start + 9)
    mid_gen = list(range(mid_start, mid_end + 1))  # 10 del medio
    last_start = max(1, max_gen - 9)
    last_gen = list(range(last_start, max_gen + 1))  # Últimas 10
    
    # Filtrar datos disponibles para cada rango
    first_data = tracked_df[tracked_df['generation'].isin(first_gen)]
    mid_data = tracked_df[tracked_df['generation'].isin(mid_gen)]
    last_data = tracked_df[tracked_df['generation'].isin(last_gen)]
    
    # Crear los 3 gráficos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Distribución de Intercambios por Períodos de Evolución', 
                 fontsize=16, fontweight='bold')
    
    # Gráfico 1: Primeras 10 generaciones
    ax1 = axes[0]
    if not first_data.empty:
        # Boxplot para cada generación
        box_data = []
        labels = []
        for gen in sorted(first_gen):
            gen_data = first_data[first_data['generation'] == gen]['n_exchanges']
            if len(gen_data) > 0:
                box_data.append(gen_data.values)
                labels.append(f'Gen {gen}')
        
        if box_data:
            bp1 = ax1.boxplot(box_data, labels=labels, patch_artist=True)
            for patch in bp1['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            # Ocultar la línea de la mediana
            for median in bp1['medians']:
                median.set_visible(False)
            # Agregar media como línea horizontal roja dentro de cada caja
            means = [np.mean(data) for data in box_data]
            positions = range(1, len(means) + 1)
            for pos, mean_val in zip(positions, means):
                ax1.plot([pos - 0.3, pos + 0.3], [mean_val, mean_val], 'r-', linewidth=2, zorder=3)
            # Agregar etiqueta solo una vez
            ax1.plot([], [], 'r-', linewidth=2, label='Media')
            ax1.set_title(f'Primeras 10 Generaciones\n(1-10)', 
                         fontsize=12, fontweight='bold')
            ax1.set_xlabel('Generación', fontsize=10)
            ax1.set_ylabel('Número de Intercambios', fontsize=10)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.legend(fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'No hay datos disponibles', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Primeras 10 Generaciones', fontsize=12)
    else:
        ax1.text(0.5, 0.5, 'No hay datos disponibles', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Primeras 10 Generaciones', fontsize=12)
    
    # Gráfico 2: 10 generaciones del medio
    ax2 = axes[1]
    if not mid_data.empty:
        box_data = []
        labels = []
        for gen in sorted(mid_gen):
            gen_data = mid_data[mid_data['generation'] == gen]['n_exchanges']
            if len(gen_data) > 0:
                box_data.append(gen_data.values)
                labels.append(f'Gen {gen}')
        
        if box_data:
            bp2 = ax2.boxplot(box_data, labels=labels, patch_artist=True)
            for patch in bp2['boxes']:
                patch.set_facecolor('lightgreen')
                patch.set_alpha(0.7)
            # Ocultar la línea de la mediana
            for median in bp2['medians']:
                median.set_visible(False)
            # Agregar media como línea horizontal roja dentro de cada caja
            means = [np.mean(data) for data in box_data]
            positions = range(1, len(means) + 1)
            for pos, mean_val in zip(positions, means):
                ax2.plot([pos - 0.3, pos + 0.3], [mean_val, mean_val], 'r-', linewidth=2, zorder=3)
            # Agregar etiqueta solo una vez
            ax2.plot([], [], 'r-', linewidth=2, label='Media')
            ax2.set_title(f'10 Generaciones del Medio\n({mid_start}-{mid_end})', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel('Generación', fontsize=10)
            ax2.set_ylabel('Número de Intercambios', fontsize=10)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No hay datos disponibles', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'10 Generaciones del Medio ({mid_start}-{mid_end})', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No hay datos disponibles', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'10 Generaciones del Medio ({mid_start}-{mid_end})', fontsize=12)
    
    # Gráfico 3: Últimas 10 generaciones
    ax3 = axes[2]
    if not last_data.empty:
        box_data = []
        labels = []
        for gen in sorted(last_gen):
            gen_data = last_data[last_data['generation'] == gen]['n_exchanges']
            if len(gen_data) > 0:
                box_data.append(gen_data.values)
                labels.append(f'Gen {gen}')
        
        if box_data:
            bp3 = ax3.boxplot(box_data, labels=labels, patch_artist=True)
            for patch in bp3['boxes']:
                patch.set_facecolor('lightcoral')
                patch.set_alpha(0.7)
            # Ocultar la línea de la mediana
            for median in bp3['medians']:
                median.set_visible(False)
            # Agregar media como línea horizontal roja dentro de cada caja
            means = [np.mean(data) for data in box_data]
            positions = range(1, len(means) + 1)
            for pos, mean_val in zip(positions, means):
                ax3.plot([pos - 0.3, pos + 0.3], [mean_val, mean_val], 'r-', linewidth=2, zorder=3)
            # Agregar etiqueta solo una vez
            ax3.plot([], [], 'r-', linewidth=2, label='Media')
            ax3.set_title(f'Últimas 10 Generaciones\n({last_start}-{max_gen})', 
                         fontsize=12, fontweight='bold')
            ax3.set_xlabel('Generación', fontsize=10)
            ax3.set_ylabel('Número de Intercambios', fontsize=10)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.legend(fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No hay datos disponibles', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'Últimas 10 Generaciones ({last_start}-{max_gen})', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'No hay datos disponibles', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'Últimas 10 Generaciones ({last_start}-{max_gen})', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "exchange_distribution_by_periods.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Gráfico de distribución por períodos guardado en: {output_path}")


def plot_pareto_front(pareto_df: pd.DataFrame, output_dir: str):
    """
    Genera gráficos del frente de Pareto en 2D
    
    Args:
        pareto_df: DataFrame con el frente de Pareto (columnas: 1-cov_health, 1-cov_education, 
                   1-cov_greens, 1-cov_work, change_ratio, solution_index, score)
        output_dir: Directorio donde guardar los gráficos
    """
    if not MATPLOTLIB_OK:
        print("[ADVERTENCIA] matplotlib no instalado: omitiendo gráfico de frente de Pareto")
        return
    
    if pareto_df.empty:
        print("[ADVERTENCIA] No hay datos del frente de Pareto para graficar")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear figura con múltiples subplots para diferentes vistas del frente de Pareto
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Análisis del Frente de Pareto', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Cobertura Salud vs Cobertura Educación
    ax1 = plt.subplot(3, 3, 1)
    scatter1 = ax1.scatter(pareto_df['1-cov_health'], pareto_df['1-cov_education'], 
                           c=pareto_df['change_ratio'], cmap='viridis', 
                           s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    # Marcar la solución óptima (menor score)
    if 'score' in pareto_df.columns:
        best_idx = pareto_df['score'].idxmin()
        ax1.scatter(pareto_df.loc[best_idx, '1-cov_health'], 
                   pareto_df.loc[best_idx, '1-cov_education'],
                   s=200, marker='*', color='red', edgecolors='black', 
                   linewidth=2, label='Solución óptima', zorder=5)
    ax1.set_xlabel('1 - Cobertura Salud', fontsize=11, fontweight='bold')
    ax1.set_ylabel('1 - Cobertura Educación', fontsize=11, fontweight='bold')
    ax1.set_title('Salud vs Educación', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter1, ax=ax1, label='Change Ratio')
    
    # Gráfico 2: Cobertura Salud vs Cobertura Áreas Verdes
    ax2 = plt.subplot(3, 3, 2)
    scatter2 = ax2.scatter(pareto_df['1-cov_health'], pareto_df['1-cov_greens'], 
                          c=pareto_df['change_ratio'], cmap='viridis', 
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    if 'score' in pareto_df.columns:
        ax2.scatter(pareto_df.loc[best_idx, '1-cov_health'], 
                   pareto_df.loc[best_idx, '1-cov_greens'],
                   s=200, marker='*', color='red', edgecolors='black', 
                   linewidth=2, label='Solución óptima', zorder=5)
    ax2.set_xlabel('1 - Cobertura Salud', fontsize=11, fontweight='bold')
    ax2.set_ylabel('1 - Cobertura Áreas Verdes', fontsize=11, fontweight='bold')
    ax2.set_title('Salud vs Áreas Verdes', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Change Ratio')
    
    # Gráfico 3: Cobertura Educación vs Cobertura Trabajo
    ax3 = plt.subplot(3, 3, 3)
    scatter3 = ax3.scatter(pareto_df['1-cov_education'], pareto_df['1-cov_work'], 
                          c=pareto_df['change_ratio'], cmap='viridis', 
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    if 'score' in pareto_df.columns:
        ax3.scatter(pareto_df.loc[best_idx, '1-cov_education'], 
                   pareto_df.loc[best_idx, '1-cov_work'],
                   s=200, marker='*', color='red', edgecolors='black', 
                   linewidth=2, label='Solución óptima', zorder=5)
    ax3.set_xlabel('1 - Cobertura Educación', fontsize=11, fontweight='bold')
    ax3.set_ylabel('1 - Cobertura Trabajo', fontsize=11, fontweight='bold')
    ax3.set_title('Educación vs Trabajo', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.colorbar(scatter3, ax=ax3, label='Change Ratio')
    
    # Gráfico 4: Change Ratio vs Déficit de Cobertura Promedio
    ax4 = plt.subplot(3, 3, 4)
    avg_coverage_deficit = (pareto_df['1-cov_health'] + pareto_df['1-cov_education'] + 
                           pareto_df['1-cov_greens'] + pareto_df['1-cov_work']) / 4.0
    scatter4 = ax4.scatter(pareto_df['change_ratio'], avg_coverage_deficit, 
                          c=pareto_df['score'] if 'score' in pareto_df.columns else None,
                          cmap='plasma', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    if 'score' in pareto_df.columns:
        ax4.scatter(pareto_df.loc[best_idx, 'change_ratio'], 
                   avg_coverage_deficit.loc[best_idx],
                   s=200, marker='*', color='red', edgecolors='black', 
                   linewidth=2, label='Solución óptima', zorder=5)
    ax4.set_xlabel('Change Ratio', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Déficit de Cobertura Promedio', fontsize=11, fontweight='bold')
    ax4.set_title('Cambio Territorial vs Déficit de Cobertura', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    if 'score' in pareto_df.columns:
        plt.colorbar(scatter4, ax=ax4, label='Score')
    
    # Gráfico 5: Distribución del Score
    ax5 = plt.subplot(3, 3, 5)
    if 'score' in pareto_df.columns:
        ax5.hist(pareto_df['score'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax5.axvline(pareto_df['score'].min(), color='red', linestyle='--', 
                   linewidth=2, label=f'Óptimo: {pareto_df["score"].min():.4f}')
        ax5.axvline(pareto_df['score'].mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'Promedio: {pareto_df["score"].mean():.4f}')
        ax5.set_xlabel('Score', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
        ax5.set_title('Distribución del Score en el Frente de Pareto', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'Score no disponible', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Distribución del Score', fontsize=12)
    
    # Gráfico 6: Cobertura vs Change Ratio (relación principal)
    ax6 = plt.subplot(3, 3, 7)
    # Calcular cobertura promedio (1 - déficit promedio)
    avg_coverage_deficit = (pareto_df['1-cov_health'] + pareto_df['1-cov_education'] + 
                           pareto_df['1-cov_greens'] + pareto_df['1-cov_work']) / 4.0
    avg_coverage = 1.0 - avg_coverage_deficit  # Convertir déficit a cobertura
    
    scatter6 = ax6.scatter(pareto_df['change_ratio'], avg_coverage * 100, 
                          c=pareto_df['score'] if 'score' in pareto_df.columns else None,
                          cmap='coolwarm', s=80, alpha=0.7, edgecolors='black', linewidth=1)
    # Marcar la solución óptima
    if 'score' in pareto_df.columns:
        best_idx = pareto_df['score'].idxmin()
        ax6.scatter(pareto_df.loc[best_idx, 'change_ratio'], 
                   avg_coverage.loc[best_idx] * 100,
                   s=300, marker='*', color='gold', edgecolors='black', 
                   linewidth=2, label='Solución óptima', zorder=5)
    ax6.set_xlabel('Change Ratio (Proporción de Cambios)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Cobertura Promedio (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Cobertura vs Cambio Territorial', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    if 'score' in pareto_df.columns:
        plt.colorbar(scatter6, ax=ax6, label='Score')
    
    # Gráfico 7: Resumen estadístico del frente
    ax7 = plt.subplot(3, 3, 8)
    ax7.axis('off')
    stats_text = f"""
    ESTADÍSTICAS DEL FRENTE DE PARETO
    
    Número de soluciones: {len(pareto_df)}
    
    Cobertura Salud:
      Min: {pareto_df['1-cov_health'].min():.3f}
      Max: {pareto_df['1-cov_health'].max():.3f}
      Promedio: {pareto_df['1-cov_health'].mean():.3f}
    
    Cobertura Educación:
      Min: {pareto_df['1-cov_education'].min():.3f}
      Max: {pareto_df['1-cov_education'].max():.3f}
      Promedio: {pareto_df['1-cov_education'].mean():.3f}
    
    Cobertura Áreas Verdes:
      Min: {pareto_df['1-cov_greens'].min():.3f}
      Max: {pareto_df['1-cov_greens'].max():.3f}
      Promedio: {pareto_df['1-cov_greens'].mean():.3f}
    
    Cobertura Trabajo:
      Min: {pareto_df['1-cov_work'].min():.3f}
      Max: {pareto_df['1-cov_work'].max():.3f}
      Promedio: {pareto_df['1-cov_work'].mean():.3f}
    
    Change Ratio:
      Min: {pareto_df['change_ratio'].min():.3f}
      Max: {pareto_df['change_ratio'].max():.3f}
      Promedio: {pareto_df['change_ratio'].mean():.3f}
    """
    if 'score' in pareto_df.columns:
        stats_text += f"""
    
    Score:
      Min: {pareto_df['score'].min():.4f}
      Max: {pareto_df['score'].max():.4f}
      Promedio: {pareto_df['score'].mean():.4f}
        """
    ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Gráfico 8: Análisis de trade-off Cobertura vs Change (gráfico adicional)
    ax8 = plt.subplot(3, 3, 9)
    # Calcular cobertura promedio para cada solución individual
    avg_coverage_deficit = (pareto_df['1-cov_health'] + pareto_df['1-cov_education'] + 
                           pareto_df['1-cov_greens'] + pareto_df['1-cov_work']) / 4.0
    avg_coverage = (1.0 - avg_coverage_deficit) * 100  # Convertir déficit a cobertura en porcentaje
    
    # Mostrar todos los puntos individuales del frente de Pareto
    scatter = ax8.scatter(pareto_df['change_ratio'], avg_coverage, 
                         c=pareto_df['score'] if 'score' in pareto_df.columns else None,
                         cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                         label='Soluciones del frente de Pareto')
    
    # Marcar la solución óptima
    if 'score' in pareto_df.columns:
        best_idx = pareto_df['score'].idxmin()
        ax8.scatter(pareto_df.loc[best_idx, 'change_ratio'], 
                   avg_coverage.loc[best_idx],
                   s=300, marker='*', color='red', edgecolors='black', 
                   linewidth=2, label='Solución óptima', zorder=5)
    
    # Si hay suficientes puntos, también mostrar línea de tendencia o promedios por rangos
    if len(pareto_df) > 10:
        # Crear bins para mostrar también promedios por rangos como referencia
        n_bins = min(10, len(pareto_df) // 3)  # Ajustar número de bins según cantidad de datos
        if n_bins > 1:
            bins = np.linspace(pareto_df['change_ratio'].min(), pareto_df['change_ratio'].max(), n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_coverage_avg = []
            
            for i in range(len(bins) - 1):
                mask = (pareto_df['change_ratio'] >= bins[i]) & (pareto_df['change_ratio'] < bins[i+1])
                if i == len(bins) - 2:
                    mask = (pareto_df['change_ratio'] >= bins[i]) & (pareto_df['change_ratio'] <= bins[i+1])
                
                if mask.sum() > 0:
                    bin_coverage_avg.append(avg_coverage.loc[mask].mean())
                else:
                    bin_coverage_avg.append(np.nan)
            
            bin_coverage_avg = np.array(bin_coverage_avg)
            valid_bins = ~np.isnan(bin_coverage_avg)
            
            if valid_bins.sum() > 1:
                ax8.plot(bin_centers[valid_bins], bin_coverage_avg[valid_bins], 
                        'r--', linewidth=2, alpha=0.5, label='Promedio por rangos')
    
    ax8.set_xlabel('Change Ratio', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Cobertura Promedio (%)', fontsize=11, fontweight='bold')
    ax8.set_title('Trade-off: Cobertura vs Cambios (Todas las Soluciones)', 
                 fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=9)
    if 'score' in pareto_df.columns:
        plt.colorbar(scatter, ax=ax8, label='Score')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "pareto_front_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Gráfico del frente de Pareto guardado en: {output_path}")


def plot_coverage_comparison(initial_metrics: Dict[str, float], 
                             final_metrics: Dict[str, float], 
                             output_dir: str):
    """
    Genera gráfico de comparación de coberturas antes y después de la optimización
    
    Args:
        initial_metrics: Diccionario con métricas iniciales (keys: cov_health, cov_education, etc.)
        final_metrics: Diccionario con métricas finales
        output_dir: Directorio donde guardar el gráfico
    """
    if not MATPLOTLIB_OK:
        print("[ADVERTENCIA] matplotlib no instalado: omitiendo gráfico de comparación")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extraer coberturas por categoría
    categories = ['health', 'education', 'greens', 'work']
    category_labels = {
        'health': 'Salud',
        'education': 'Educación',
        'greens': 'Áreas Verdes',
        'work': 'Trabajo'
    }
    
    initial_covs = []
    final_covs = []
    improvements = []
    labels = []
    
    for cat in categories:
        key_initial = f'cov_{cat}'
        key_final = f'cov_{cat}'
        
        if key_initial in initial_metrics and key_final in final_metrics:
            init_val = initial_metrics[key_initial]
            final_val = final_metrics[key_final]
            improvement = final_val - init_val
            
            initial_covs.append(init_val)
            final_covs.append(final_val)
            improvements.append(improvement)
            labels.append(category_labels[cat])
    
    # También agregar cobertura integral (todas las categorías)
    if 'cov_all' in initial_metrics and 'cov_all' in final_metrics:
        initial_covs.append(initial_metrics['cov_all'])
        final_covs.append(final_metrics['cov_all'])
        improvements.append(final_metrics['cov_all'] - initial_metrics['cov_all'])
        labels.append('Todas las\nCategorías')
    
    if not initial_covs:
        print("[ADVERTENCIA] No hay datos de cobertura para comparar")
        return
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comparación de Coberturas: Estado Inicial vs Optimizado', 
                fontsize=16, fontweight='bold')
    
    # Gráfico 1: Barras comparativas
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, [v * 100 for v in initial_covs], width, 
                   label='Estado Inicial', color='#ff6b6b', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, [v * 100 for v in final_covs], width, 
                   label='Estado Optimizado', color='#51cf66', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Agregar valores en las barras
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{height1:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{height2:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Categoría de Servicio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cobertura (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Cobertura por Categoría', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(max(initial_covs), max(final_covs)) * 100 * 1.15)
    
    # Gráfico 2: Mejora porcentual
    colors = ['#4dabf7' if imp >= 0 else '#ff8787' for imp in improvements]
    bars3 = ax2.barh(labels, [imp * 100 for imp in improvements], 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Agregar valores en las barras
    for i, (bar, imp) in enumerate(zip(bars3, improvements)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:+.1f}%', ha='left' if width >= 0 else 'right', 
                va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Mejora en Cobertura (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Mejora por Categoría', fontsize=13, fontweight='bold')
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "coverage_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Gráfico de comparación de coberturas guardado en: {output_path}")


def run_reordering_optimization(
    G: nx.MultiDiGraph,
    homes: gpd.GeoDataFrame,
    services: Dict[str, gpd.GeoDataFrame],
    target_category: str,
    minutes: float = 15.0,
    max_gen: int = 200,
    pop_size: int = 100,
    alpha_balance: float = 0.1
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, float]:
    """
    Ejecuta optimización con reordenamiento
    
    Returns:
        (nuevos_hogares, nuevos_servicios, frente_pareto, mejor_cobertura)
    """
    if not PYMOO_OK:
        raise RuntimeError("pymoo no está instalado")
    
    print(f"\n[NSGA-II] Iniciando optimización con reordenamiento para: {target_category}")
    print(f"  Generaciones: {max_gen}, Población: {pop_size}")
    
    problem = ReorderingProblem(
        G, homes, services, target_category, minutes, alpha_balance
    )
    
    # Usar inicialización factible y reparador
    sampling = FeasibleSampling(n_homes=problem.n_homes)
    repair = FeasibleRepair(n_homes=problem.n_homes)
    algorithm = NSGA2(pop_size=pop_size, sampling=sampling, repair=repair)
    termination = get_termination("n_gen", max_gen)
    
    res = minimize(problem, algorithm, termination, verbose=True, seed=42)
    
    # Analizar frente de Pareto
    F = res.F
    X = res.X
    
    # Verificar que tenemos resultados
    if X is None or F is None or len(X) == 0:
        print("[ERROR] La optimización no produjo resultados válidos")
        print(f"  res.X: {X}")
        print(f"  res.F: {F}")
        # Retornar configuración inicial como fallback
        new_homes = homes.copy()
        new_services = services.get(target_category, gpd.GeoDataFrame(geometry=[], crs=4326)).copy()
        if new_services.empty:
            new_services = gpd.GeoDataFrame(geometry=[], crs=4326)
        pareto = pd.DataFrame({"1-coverage": [1.0], "balance_penalty": [0.0], "solution_index": [0]})
        best_cov = 0.0
        return new_homes, new_services, pareto, best_cov
    
    # Filtrar solo soluciones factibles (que cumplen restricción de n_homes)
    # Usamos un umbral más permisivo que coincide con el margen en la restricción
    margin = max(1, int(problem.n_homes * 0.01))
    feasible_mask = []
    for x in X:
        n_homes_actual = int((x == 0).sum())
        # Considerar factible si está dentro del margen permitido
        feasible_mask.append(abs(n_homes_actual - problem.n_homes) <= margin)
    feasible_mask = np.array(feasible_mask)
    
    if not np.any(feasible_mask):
        print(f"[ADVERTENCIA] No se encontraron soluciones factibles (margen: ±{margin})")
        print(f"  Usando todas las soluciones disponibles")
        feasible_mask = np.ones(len(X), dtype=bool)
    else:
        print(f"[INFO] {feasible_mask.sum()}/{len(X)} soluciones factibles encontradas")
    
    F_feas = F[feasible_mask]
    X_feas = X[feasible_mask]
    
    pareto = pd.DataFrame({
        "1-coverage": F_feas[:, 0],
        "balance_penalty": F_feas[:, 1]
    })
    pareto["solution_index"] = np.arange(len(pareto))
    
    # Elegir mejor solución (equilibrio entre cobertura y balance)
    norm = (pareto - pareto.min()) / (pareto.max() - pareto.min() + 1e-9)
    pareto["score"] = norm["1-coverage"] + 0.3 * norm["balance_penalty"]
    
    best_idx = int(pareto.sort_values("score").iloc[0]["solution_index"])
    x_best = X_feas[best_idx]
    
    # Reconstruir configuración óptima
    home_mask = (x_best == 0)
    service_mask = (x_best == 1)
    
    new_homes = problem.location_pool[home_mask].copy()
    new_homes["category"] = "home"
    new_homes["type"] = "home"
    new_homes["iteration"] = "optimized"
    
    new_services = problem.location_pool[service_mask].copy()
    new_services["category"] = target_category
    new_services["type"] = "service"
    new_services["iteration"] = "optimized"
    
    best_cov = 1.0 - float(F_feas[best_idx, 0])
    
    print(f"\n[Resultado] Mejor cobertura: {best_cov:.3f}")
    print(f"  Hogares: {len(new_homes)} (objetivo: {problem.n_homes})")
    print(f"  Servicios ({target_category}): {len(new_services)}")
    
    return new_homes, new_services, pareto, best_cov


def run_reordering_optimization_all_categories(
    G: nx.MultiDiGraph,
    homes: gpd.GeoDataFrame,
    services: Dict[str, gpd.GeoDataFrame],
    minutes: float = 15.0,
    max_gen: int = 200,
    pop_size: int = 50,
    callback: Optional[EvolutionCallback] = None,
    track_generations: List[int] = None
) -> Tuple[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame], pd.DataFrame, Dict[str, float], Optional[EvolutionCallback]]:
    """
    Ejecuta optimización con reordenamiento para TODAS las categorías simultáneamente
    
    Args:
        callback: Callback personalizado para tracking (si None, se crea uno nuevo)
        track_generations: Lista de generaciones específicas a capturar (ej: [1, 2, 3, 78, 79, 80])
    
    Returns:
        (nuevos_hogares, nuevos_servicios_por_categoria, frente_pareto, mejores_coberturas, callback)
    """
    if not PYMOO_OK:
        raise RuntimeError("pymoo no está instalado")
    
    print(f"\n[NSGA-II] Iniciando optimización con reordenamiento para TODAS las categorías")
    print(f"  Generaciones: {max_gen}, Población: {pop_size}")
    
    problem = ReorderingProblemAllCategories(
        G, homes, services, minutes
    )
    
    # Crear callback si no se proporciona uno
    if callback is None:
        initial_config = problem.initial_config
        callback = EvolutionCallback(initial_config, track_generations=track_generations)
        print(f"  [Tracking] Callback creado para rastrear intercambios")
        if track_generations:
            print(f"  [Tracking] Generaciones específicas a capturar: {track_generations}")
    
    # Usar inicialización factible, operadores personalizados y reparador para todas las categorías
    # initial_change_percentage=0.02: 2% de cambios iniciales respecto a configuración inicial
    # Con 3200 puntos: ~64 cambios iniciales (en lugar de ~2000)
    sampling = FeasibleSamplingAllCategories(
        n_homes=problem.n_homes,
        n_health=problem.n_health,
        n_education=problem.n_education,
        n_greens=problem.n_greens,
        n_work=problem.n_work,
        initial_change_percentage=0.02  # 2% de cambios iniciales
    )
    crossover = FeasibleCrossoverAllCategories(
        n_homes=problem.n_homes,
        n_health=problem.n_health,
        n_education=problem.n_education,
        n_greens=problem.n_greens,
        n_work=problem.n_work,
        prob=0.9
    )
    mutation = FeasibleMutationAllCategories(
        n_homes=problem.n_homes,
        n_health=problem.n_health,
        n_education=problem.n_education,
        n_greens=problem.n_greens,
        n_work=problem.n_work,
        prob=0.7
    )
    repair = FeasibleRepairAllCategories(
        n_homes=problem.n_homes,
        n_health=problem.n_health,
        n_education=problem.n_education,
        n_greens=problem.n_greens,
        n_work=problem.n_work
    )
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        repair=repair,
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", max_gen)
    
    # Pasar callback a minimize solo si existe (en pymoo 0.6.1.1)
    if callback is not None:
        res = minimize(problem, algorithm, termination, verbose=True, seed=42, callback=callback)
    else:
        res = minimize(problem, algorithm, termination, verbose=True, seed=42)
    
    # Analizar frente de Pareto
    F = res.F
    X = res.X
    
    # Verificar que tenemos resultados
    if X is None or F is None or len(X) == 0:
        print("[ERROR] La optimización no produjo resultados válidos")
        print(f"  res.X: {X}")
        print(f"  res.F: {F}")
        # Retornar configuración inicial como fallback
        new_homes = homes.copy()
        new_services = {k: v.copy() for k, v in services.items()}
        pareto = pd.DataFrame()
        best_covs = {cat: 0.0 for cat in services.keys()}
        return new_homes, new_services, pareto, best_covs, callback
    
    # Filtrar soluciones factibles
    margin = max(1, int(min(problem.n_homes, problem.n_health, problem.n_education, problem.n_greens, problem.n_work) * 0.01))
    feasible_mask = []
    for x in X:
        n_homes_actual = int((x == 0).sum())
        n_health_actual = int((x == 1).sum())
        n_education_actual = int((x == 2).sum())
        n_greens_actual = int((x == 3).sum())
        n_work_actual = int((x == 4).sum())
        
        feasible = (
            abs(n_homes_actual - problem.n_homes) <= margin and
            abs(n_health_actual - problem.n_health) <= margin and
            abs(n_education_actual - problem.n_education) <= margin and
            abs(n_greens_actual - problem.n_greens) <= margin and
            abs(n_work_actual - problem.n_work) <= margin
        )
        feasible_mask.append(feasible)
    feasible_mask = np.array(feasible_mask)
    
    if not np.any(feasible_mask):
        print(f"[ADVERTENCIA] No se encontraron soluciones factibles (margen: ±{margin})")
        print(f"  Usando todas las soluciones disponibles")
        feasible_mask = np.ones(len(X), dtype=bool)
    else:
        print(f"[INFO] {feasible_mask.sum()}/{len(X)} soluciones factibles encontradas")
    
    F_feas = F[feasible_mask]
    X_feas = X[feasible_mask]
    
    # Crear DataFrame del frente de Pareto
    # Nota: F_feas[:, 4] contiene change_ratio * 5.0, necesitamos dividir para obtener el ratio puro
    change_ratio_raw = F_feas[:, 4] / 5.0  # Dividir por 5.0 para obtener el ratio puro
    
    pareto = pd.DataFrame({
        "1-cov_health": F_feas[:, 0],
        "1-cov_education": F_feas[:, 1],
        "1-cov_greens": F_feas[:, 2],
        "1-cov_work": F_feas[:, 3],
        "change_ratio": change_ratio_raw  # Proporción de cambios (sin peso)
    })
    pareto["solution_index"] = np.arange(len(pareto))
    
    # Elegir mejor solución (balance entre cobertura y minimización de cambios)
    # Normalizar objetivos (0-1) y balancear con peso moderado en cambios
    norm = (pareto.iloc[:, :4] - pareto.iloc[:, :4].min()) / (pareto.iloc[:, :4].max() - pareto.iloc[:, :4].min() + 1e-9)
    # Normalizar change_ratio por separado
    norm_changes = pareto["change_ratio"] / (pareto["change_ratio"].max() + 1e-9)
    # Minimizar suma de coberturas (menor es mejor) y cambios (menor es mejor)
    # Peso 5.0 para cambios: PENALIZAR moderadamente los cambios en la selección final
    # Prioriza soluciones con menos cambios, pero permite más exploración que con pesos extremos.
    # Con este peso, elegirá soluciones que balanceen cobertura y cambios razonablemente.
    pareto["score"] = norm.sum(axis=1) + 5.0 * norm_changes
    
    best_idx = int(pareto.sort_values("score").iloc[0]["solution_index"])
    x_best = X_feas[best_idx]
    
    # Reconstruir configuración óptima
    new_homes = problem.location_pool[x_best == 0].copy()
    new_homes["category"] = "home"
    new_homes["type"] = "home"
    new_homes["iteration"] = "optimized"
    
    new_services = {}
    new_services["health"] = problem.location_pool[x_best == 1].copy()
    new_services["health"]["category"] = "health"
    new_services["health"]["type"] = "service"
    new_services["health"]["iteration"] = "optimized"
    
    new_services["education"] = problem.location_pool[x_best == 2].copy()
    new_services["education"]["category"] = "education"
    new_services["education"]["type"] = "service"
    new_services["education"]["iteration"] = "optimized"
    
    new_services["greens"] = problem.location_pool[x_best == 3].copy()
    new_services["greens"]["category"] = "greens"
    new_services["greens"]["type"] = "service"
    new_services["greens"]["iteration"] = "optimized"
    
    new_services["work"] = problem.location_pool[x_best == 4].copy()
    new_services["work"]["category"] = "work"
    new_services["work"]["type"] = "service"
    new_services["work"]["iteration"] = "optimized"
    
    best_covs = {
        "health": 1.0 - float(F_feas[best_idx, 0]),
        "education": 1.0 - float(F_feas[best_idx, 1]),
        "greens": 1.0 - float(F_feas[best_idx, 2]),
        "work": 1.0 - float(F_feas[best_idx, 3])
    }
    
    # Calcular número de cambios en la mejor solución
    n_changes = int((x_best != problem.initial_config).sum())
    total_locations = len(x_best)
    change_percentage = (n_changes / total_locations * 100) if total_locations > 0 else 0.0
    
    print(f"\n[Resultado] Mejores coberturas:")
    for cat, cov in best_covs.items():
        print(f"  {cat}: {cov:.3f}")
    print(f"  Hogares: {len(new_homes)} (objetivo: {problem.n_homes})")
    print(f"  Cambios realizados: {n_changes}/{total_locations} ({change_percentage:.1f}%)")
    
    # Mostrar resumen de tracking si existe
    if callback and callback.evolution_history:
        stats_df = callback.get_exchange_stats()
        if not stats_df.empty:
            print(f"\n[Tracking] Estadísticas de intercambios:")
            print(f"  Generaciones rastreadas: {len(callback.generation_data)}")
            print(f"  Intercambios promedio (inicial): {stats_df.iloc[0]['mean_exchanges']:.1f}")
            print(f"  Intercambios promedio (final): {stats_df.iloc[-1]['mean_exchanges']:.1f}")
            if len(stats_df) > 1:
                improvement = stats_df.iloc[-1]['mean_exchanges'] - stats_df.iloc[0]['mean_exchanges']
                print(f"  Evolución de intercambios: {improvement:+.1f}")
    
    return new_homes, new_services, pareto, best_covs, callback


# -----------------------------
# OPTIMIZACIÓN ITERATIVA
# -----------------------------

def iterative_reordering(
    G: nx.MultiDiGraph,
    initial_homes: gpd.GeoDataFrame,
    initial_services: Dict[str, gpd.GeoDataFrame],
    categories: List[str],
    minutes: float = 15.0,
    n_iterations: int = 1,
    max_gen: int = 200,
    pop_size: int = 50,
    track_generations: List[int] = None,
    output_dir: str = None
) -> Tuple[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame], List[Dict], Optional[EvolutionCallback], pd.DataFrame]:
    """
    Ejecuta optimización de TODAS las categorías simultáneamente (una sola iteración)
    
    Args:
        track_generations: Lista de generaciones específicas a capturar (ej: [1, 2, 3, 78, 79, 80])
        output_dir: Directorio donde exportar estadísticas y gráficos
    
    Returns:
        (hogares_finales, servicios_finales_por_categoria, historial_metricas, callback, pareto_df)
    """
    # Forzar a 1 iteración (todas las categorías juntas)
    n_iterations = 1
    
    # Configurar generaciones a rastrear por defecto
    if track_generations is None:
        if max_gen >= 80:
            track_generations = [1, 2, 3, max_gen-2, max_gen-1, max_gen]
        else:
            track_generations = [1, 2, 3, max_gen-2, max_gen-1, max_gen]
    
    print("\n" + "="*70)
    print("OPTIMIZACIÓN CON REORDENAMIENTO - TODAS LAS CATEGORÍAS JUNTAS")
    print("="*70)
    print(f"\n[Tracking] Generaciones específicas a capturar: {track_generations}")
    
    history = []
    
    # Evaluación inicial
    _, initial_metrics = evaluate_all_categories(G, initial_homes, initial_services, minutes)
    history.append({
        "iteration": 0,
        "category": "initial",
        **initial_metrics
    })
    
    print(f"\n[Estado Inicial]")
    for k, v in initial_metrics.items():
        print(f"  {k}: {v:.3f}")
    
    # Optimización de todas las categorías juntas (una sola iteración)
    print(f"\n{'='*70}")
    print(f"OPTIMIZANDO TODAS LAS CATEGORÍAS SIMULTÁNEAMENTE")
    print(f"{'='*70}")
    
    final_homes, final_services, pareto_df, best_covs, callback = run_reordering_optimization_all_categories(
        G=G,
        homes=initial_homes,
        services=initial_services,
        minutes=minutes,
        max_gen=max_gen,
        pop_size=pop_size,
        track_generations=track_generations
    )
    
    # Evaluar estado final
    _, final_metrics = evaluate_all_categories(G, final_homes, final_services, minutes)
    
    history.append({
        "iteration": 1,
        "category": "all_categories",
        **final_metrics
    })
    
    print(f"\n[Métricas después de optimización]")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.3f}")
    
    print("\n" + "="*70)
    print("OPTIMIZACIÓN COMPLETADA")
    print("="*70)
    
    # Resumen de mejoras
    initial_cov_all = initial_metrics["cov_all"]
    final_cov_all = final_metrics["cov_all"]
    improvement = ((final_cov_all - initial_cov_all) / max(initial_cov_all, 0.001)) * 100
    
    print(f"\n[RESUMEN]")
    print(f"  Cobertura inicial (todas las categorías): {initial_cov_all:.3f}")
    print(f"  Cobertura final (todas las categorías): {final_cov_all:.3f}")
    print(f"  Mejora: {improvement:+.1f}%")
    print(f"  Hogares mantenidos: {len(final_homes)} (inicial: {len(initial_homes)})")
    
    # Exportar estadísticas y gráficos si hay callback y directorio de salida
    if callback and output_dir:
        print(f"\n[Exportando estadísticas y gráficos evolutivos...]")
        callback.export_detailed_stats(output_dir)
        plot_exchange_evolution(callback, output_dir)
        # Gráficos de distribución por períodos (primeras 10, medio 10, últimas 10)
        plot_distribution_by_periods(callback, output_dir, max_gen=max_gen)
        print(f"  Estadísticas y gráficos guardados en: {output_dir}")
    
    return final_homes, final_services, history, callback, pareto_df


# -----------------------------
# VISUALIZACIÓN
# -----------------------------

try:
    import folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False


def create_initial_map_with_nodes(
    boundary: gpd.GeoDataFrame,
    homes: gpd.GeoDataFrame,
    services: Dict[str, gpd.GeoDataFrame],
    G: nx.MultiDiGraph,
    minutes: float = 15.0
):
    """Crea mapa interactivo del estado inicial donde al hacer clic en una casa se muestra su nodo más cercano"""
    if not FOLIUM_OK:
        print("folium no instalado: omitiendo mapa")
        return None
    
    import folium
    from folium import plugins
    
    # Calcular nodos más cercanos para cada casa
    print("  [Mapa Inicial] Calculando nodos más cercanos para cada casa...")
    home_nodes = nearest_node_series(G, homes)
    
    # Obtener coordenadas de los nodos
    node_coords = {}
    for idx, home_node in home_nodes.items():
        if home_node is not None and home_node in G.nodes:
            node_coords[idx] = {
                'node_id': home_node,
                'lat': G.nodes[home_node].get('y', homes.loc[idx].geometry.y),
                'lon': G.nodes[home_node].get('x', homes.loc[idx].geometry.x)
            }
    
    center = [boundary.geometry.centroid.y.iloc[0], boundary.geometry.centroid.x.iloc[0]]
    m = folium.Map(location=center, zoom_start=14, control_scale=True)
    
    # Límite
    folium.GeoJson(
        boundary.to_json(),
        name="Límite del distrito",
        style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 2}
    ).add_to(m)
    
    # Grupo para nodos (inicialmente oculto)
    fg_nodes = folium.FeatureGroup(name="🔵 Nodos de la Red", show=False).add_to(m)
    
    # Marcar nodos únicos usados por las casas
    unique_nodes = {}
    for idx, node_info in node_coords.items():
        node_id = node_info['node_id']
        if node_id not in unique_nodes:
            unique_nodes[node_id] = node_info
    
    # Agregar nodos al mapa
    for node_id, node_info in unique_nodes.items():
        folium.CircleMarker(
            [node_info['lat'], node_info['lon']],
            radius=4,
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.6,
            weight=2,
            tooltip=f"Nodo ID: {node_id}"
        ).add_to(fg_nodes)
    
    # Grupo para casas
    fg_homes = folium.FeatureGroup(name="🏠 Hogares (Click para ver nodo)", show=True).add_to(m)
    
    # Agregar casas con popup interactivo
    for idx, row in homes.iterrows():
        node_info = node_coords.get(idx, None)
        
        if node_info:
            # Popup con información del nodo
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <h4>🏠 Hogar #{idx}</h4>
                <hr>
                <p><strong>Ubicación:</strong></p>
                <p>Lat: {row.geometry.y:.6f}<br>Lon: {row.geometry.x:.6f}</p>
                <hr>
                <p><strong>🔵 Nodo más cercano:</strong></p>
                <p><b>ID del nodo:</b> {node_info['node_id']}</p>
                <p><b>Lat:</b> {node_info['lat']:.6f}</p>
                <p><b>Lon:</b> {node_info['lon']:.6f}</p>
            </div>
            """
        else:
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <h4>🏠 Hogar #{idx}</h4>
                <hr>
                <p><strong>Ubicación:</strong></p>
                <p>Lat: {row.geometry.y:.6f}<br>Lon: {row.geometry.x:.6f}</p>
                <hr>
                <p><strong>⚠️ Nodo no encontrado</strong></p>
            </div>
            """
        
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x],
            radius=3,
            color='darkblue',
            fill=True,
            fillColor='darkblue',
            fillOpacity=0.7,
            weight=2,
            tooltip=f"Hogar #{idx} - Click para ver nodo",
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(fg_homes)
    
    # Servicios iniciales
    fg_services = folium.FeatureGroup(name="📍 Servicios", show=True).add_to(m)
    colors = {"health": "red", "education": "blue", "greens": "green", "work": "purple"}
    for cat, g in services.items():
        for idx, row in g.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=6,
                color=colors.get(cat, 'gray'),
                fill=True,
                fillColor=colors.get(cat, 'gray'),
                fillOpacity=0.8,
                weight=2,
                tooltip=f"Servicio: {cat}"
            ).add_to(fg_services)
    
    # Leyenda
    legend_html = f'''
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 280px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <p><strong>🗺️ Mapa Estado Inicial (Antes de NSGA-II)</strong></p>
        <hr>
        <p><span style="color:darkblue">●</span> <strong>Hogar</strong> (Click para ver nodo)</p>
        <p><span style="color:blue">●</span> Nodo de la red más cercano</p>
        <hr>
        <p><span style="color:red">●</span> Salud</p>
        <p><span style="color:blue">●</span> Educación</p>
        <p><span style="color:green">●</span> Áreas verdes</p>
        <p><span style="color:purple">●</span> Trabajo</p>
        <hr>
        <p><small><em>Click en una casa para ver su nodo más cercano</em></small></p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m


def create_comparison_map(
    boundary: gpd.GeoDataFrame,
    initial_homes: gpd.GeoDataFrame,
    initial_services: Dict[str, gpd.GeoDataFrame],
    final_homes: gpd.GeoDataFrame,
    final_services: Dict[str, gpd.GeoDataFrame],
    initial_reach: pd.DataFrame,
    final_reach: pd.DataFrame,
    minutes: float = 15.0
):
    """Crea mapa comparativo con estado inicial y final"""
    if not FOLIUM_OK:
        print("folium no instalado: omitiendo mapa")
        return None
    
    import folium
    from folium import plugins
    
    center = [boundary.geometry.centroid.y.iloc[0], boundary.geometry.centroid.x.iloc[0]]
    m = folium.Map(location=center, zoom_start=14, control_scale=True)
    
    # Límite
    folium.GeoJson(
        boundary.to_json(),
        name="Límite del distrito",
        style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 2}
    ).add_to(m)
    
    # ESTADO INICIAL
    fg_initial = folium.FeatureGroup(name="🔴 Estado Inicial", show=True).add_to(m)
    
    # Hogares iniciales cubiertos/no cubiertos
    if initial_reach is not None:
        covered_init = initial_homes[initial_reach["all_categories"]]
        uncovered_init = initial_homes[~initial_reach["all_categories"]]
        
        for _, row in covered_init.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=2,
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.4,
                tooltip="Hogar inicial: cubierto"
            ).add_to(fg_initial)
        
        for _, row in uncovered_init.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=2,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.4,
                tooltip="Hogar inicial: NO cubierto"
            ).add_to(fg_initial)
    
    # Servicios iniciales
    colors_init = {"health": "darkred", "education": "darkblue", "greens": "darkgreen", "work": "purple"}
    for cat, g in initial_services.items():
        for _, row in g.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=5,
                color=colors_init.get(cat, 'gray'),
                fill=True,
                fillColor=colors_init.get(cat, 'gray'),
                fillOpacity=0.7,
                tooltip=f"Servicio inicial: {cat}"
            ).add_to(fg_initial)
    
    # ESTADO FINAL
    fg_final = folium.FeatureGroup(name="🟢 Estado Optimizado", show=True).add_to(m)
    
    # Hogares finales cubiertos/no cubiertos
    if final_reach is not None:
        covered_final = final_homes[final_reach["all_categories"]]
        uncovered_final = final_homes[~final_reach["all_categories"]]
        
        for _, row in covered_final.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=3,
                color='lime',
                fill=True,
                fillColor='lime',
                fillOpacity=0.7,
                weight=2,
                tooltip="Hogar optimizado: cubierto"
            ).add_to(fg_final)
        
        for _, row in uncovered_final.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=3,
                color='orange',
                fill=True,
                fillColor='orange',
                fillOpacity=0.7,
                weight=2,
                tooltip="Hogar optimizado: NO cubierto"
            ).add_to(fg_final)
    
    # Servicios finales
    colors_final = {"health": "red", "education": "blue", "greens": "green", "work": "purple"}
    for cat, g in final_services.items():
        for _, row in g.iterrows():
            folium.CircleMarker(
                [row.geometry.y, row.geometry.x],
                radius=6,
                color=colors_final.get(cat, 'gray'),
                fill=True,
                fillColor=colors_final.get(cat, 'gray'),
                fillOpacity=0.9,
                weight=2,
                tooltip=f"Servicio optimizado: {cat}"
            ).add_to(fg_final)
    
    # Leyenda
    legend_html = f'''
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 250px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <p><strong>Ciudad de {minutes} Minutos</strong></p>
        <p><span style="color:green">●</span> Hogar cubierto (inicial)</p>
        <p><span style="color:red">●</span> Hogar NO cubierto (inicial)</p>
        <p><span style="color:lime">●</span> Hogar cubierto (optimizado)</p>
        <p><span style="color:orange">●</span> Hogar NO cubierto (optimizado)</p>
        <hr>
        <p><span style="color:darkred">●</span> Salud (inicial)</p>
        <p><span style="color:darkblue">●</span> Educación (inicial)</p>
        <p><span style="color:darkgreen">●</span> Áreas verdes (inicial)</p>
        <p><span style="color:red">◉</span> Salud (optimizado)</p>
        <p><span style="color:blue">◉</span> Educación (optimizado)</p>
        <p><span style="color:green">◉</span> Áreas verdes (optimizado)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m


# -----------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Planificación Urbana con Reordenamiento Dinámico"
    )
    parser.add_argument("--place", type=str, required=True, 
                       help="Lugar (ej: 'San Juan de Miraflores, Lima, Peru')")
    parser.add_argument("--minutes", type=float, default=15.0,
                       help="Umbral de minutos para accesibilidad")
    parser.add_argument("--speed-kmh", type=float, default=4.5,
                       help="Velocidad peatonal en km/h")
    parser.add_argument("--max-homes", type=int, default=None,
                       help="Número máximo de hogares a considerar (None = todos los encontrados)")
    parser.add_argument("--iterations", type=int, default=1,
                       help="Número de iteraciones de optimización (ahora solo 1: todas las categorías juntas)")
    parser.add_argument("--generations", type=int, default=200,
                       help="Generaciones por optimización NSGA-II")
    parser.add_argument("--population", type=int, default=50,
                       help="Tamaño de población NSGA-II")
    parser.add_argument("--categories", type=str, nargs='+',
                       default=["health", "education", "greens", "work"],
                       help="Categorías a optimizar")
    parser.add_argument("--plot", action="store_true",
                       help="Generar mapa interactivo")
    parser.add_argument("--output-dir", type=str, default="outputs_reordenamiento",
                       help="Directorio de salida")
    parser.add_argument("--track-generations", type=int, nargs='+', default=None,
                       help="Generaciones específicas a capturar para tracking (ej: 1 2 3 78 79 80)")
    
    args = parser.parse_args()
    
    # Configurar generaciones a rastrear
    track_generations = args.track_generations
    if track_generations is None:
        # Por defecto: primeras 10, 10 del medio, y últimas 10 generaciones
        max_gen = args.generations
        # Primeras 10: 1-10
        # 10 del medio: aproximadamente la mitad ± 5
        mid_start = max(1, (max_gen // 2) - 4)
        mid_end = min(max_gen, mid_start + 9)
        # Últimas 10: max_gen-9 a max_gen
        last_start = max(1, max_gen - 9)
        track_generations = list(range(1, 11)) + list(range(mid_start, mid_end + 1)) + list(range(last_start, max_gen + 1))
    
    # Crear directorio de salida
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SISTEMA DE PLANIFICACIÓN URBANA CON REORDENAMIENTO")
    print("="*70)
    print(f"\n[Configuración]")
    print(f"  Lugar: {args.place}")
    print(f"  Umbral: {args.minutes} minutos")
    print(f"  Iteraciones: {args.iterations}")
    print(f"  Categorías: {', '.join(args.categories)}")
    print(f"  Directorio de salida: {out_dir}")
    
    # 1. CARGAR DATOS
    print(f"\n[1/5] Cargando datos geográficos...")
    boundary = load_place_boundary(args.place)
    
    print(f"[2/5] Cargando red peatonal...")
    G = load_walking_graph(boundary, speed_kmh=args.speed_kmh)
    print(f"  Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}")
    
    print(f"[3/5] Cargando servicios...")
    services = load_services(boundary)
    for cat, gdf in services.items():
        print(f"  {cat}: {len(gdf)} puntos")
    
    print(f"[4/5] Cargando hogares...")
    homes = load_residences(boundary, max_points=args.max_homes)
    if args.max_homes is None:
        print(f"  Hogares: {len(homes)} (todos los encontrados en el mapa)")
    else:
        print(f"  Hogares: {len(homes)} (límite: {args.max_homes})")
    
    # 2. EVALUACIÓN INICIAL
    print(f"\n[5/5] Evaluando estado inicial...")
    initial_reach, initial_metrics = evaluate_all_categories(
        G, homes, services, args.minutes
    )
    
    print("\n[ESTADO INICIAL - Métricas de Cobertura]")
    for k, v in initial_metrics.items():
        print(f"  {k}: {v:.3f} ({v*100:.1f}%)")
    
    # 2.5. GENERAR MAPA INICIAL (ANTES DE NSGA-II)
    if FOLIUM_OK:
        print(f"\n[Generando mapa inicial con nodos...]")
        m_initial = create_initial_map_with_nodes(
            boundary, homes, services, G, args.minutes
        )
        if m_initial is not None:
            initial_map_path = os.path.join(out_dir, "initial_map_with_nodes.html")
            m_initial.save(initial_map_path)
            print(f"  Mapa inicial guardado en: {initial_map_path}")
            print(f"  (Click en una casa para ver su nodo más cercano)")
    
    # 3. OPTIMIZACIÓN ITERATIVA
    final_homes, final_services, history, callback, pareto_df = iterative_reordering(
        G=G,
        initial_homes=homes,
        initial_services=services,
        categories=args.categories,
        minutes=args.minutes,
        n_iterations=args.iterations,
        max_gen=args.generations,
        pop_size=args.population,
        track_generations=track_generations,
        output_dir=out_dir
    )
    
    # 4. EVALUACIÓN FINAL
    final_reach, final_metrics = evaluate_all_categories(
        G, final_homes, final_services, args.minutes
    )
    
    # 5. GUARDAR RESULTADOS
    print(f"\n[Guardando resultados en: {out_dir}]")
    
    # Hogares
    homes_initial = homes.copy()
    homes_initial["covered_all"] = initial_reach["all_categories"].values
    homes_initial["state"] = "initial"
    homes_initial.to_file(os.path.join(out_dir, "homes_initial.geojson"), driver="GeoJSON")
    
    final_homes_out = final_homes.copy()
    final_homes_out["covered_all"] = final_reach["all_categories"].values
    final_homes_out["state"] = "optimized"
    final_homes_out.to_file(os.path.join(out_dir, "homes_optimized.geojson"), driver="GeoJSON")
    
    # Servicios iniciales
    for cat, g in services.items():
        g_out = g.copy()
        g_out["state"] = "initial"
        g_out.to_file(os.path.join(out_dir, f"services_{cat}_initial.geojson"), driver="GeoJSON")
    
    # Servicios optimizados
    for cat, g in final_services.items():
        g_out = g.copy()
        g_out["state"] = "optimized"
        g_out.to_file(os.path.join(out_dir, f"services_{cat}_optimized.geojson"), driver="GeoJSON")
    
    # Historial de métricas
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "optimization_history.csv"), index=False)
    
    # Comparativa
    comparison = pd.DataFrame({
        "metric": list(initial_metrics.keys()),
        "initial": list(initial_metrics.values()),
        "final": list(final_metrics.values())
    })
    comparison["improvement"] = comparison["final"] - comparison["initial"]
    comparison["improvement_pct"] = (comparison["improvement"] / comparison["initial"].clip(lower=0.001)) * 100
    comparison.to_csv(os.path.join(out_dir, "comparison_metrics.csv"), index=False)
    
    print("\n[COMPARATIVA FINAL]")
    print(comparison.to_string(index=False))
    
    # 6. GENERAR GRÁFICOS ADICIONALES
    if MATPLOTLIB_OK:
        print(f"\n[Generando gráficos adicionales...]")
        
        # Gráfico del frente de Pareto
        if pareto_df is not None and not pareto_df.empty:
            plot_pareto_front(pareto_df, out_dir)
        else:
            print("  [ADVERTENCIA] No hay datos del frente de Pareto para graficar")
        
        # Gráfico de comparación de coberturas
        plot_coverage_comparison(initial_metrics, final_metrics, out_dir)
    
    # 7. GENERAR MAPA
    if FOLIUM_OK:
        print(f"\n[Generando mapa comparativo...]")
        m = create_comparison_map(
            boundary, homes, services,
            final_homes, final_services,
            initial_reach, final_reach,
            args.minutes
        )
        if m is not None:
            map_path = os.path.join(out_dir, "comparison_map.html")
            m.save(map_path)
            print(f"  Mapa guardado en: {map_path}")
    elif args.plot:
        print(f"\n[ADVERTENCIA] folium no está instalado. Instala con: pip install folium")
    
    print("\n" + "="*70)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\nTodos los archivos se guardaron en: {out_dir}")


if __name__ == "__main__":
    main()
