"""
Sistema de Planificación Urbana con Reordenamiento Dinámico
Autor: Mejorado por Claude
Descripción:
 Este sistema permite el INTERCAMBIO DINÁMICO entre hogares y servicios para optimizar
 la accesibilidad urbana bajo el concepto de Ciudad de 15 Minutos.
 
 CARACTERÍSTICAS PRINCIPALES:
 1. Mantiene constante el número de hogares
 2. Permite que hogares y servicios intercambien posiciones
 3. Optimización iterativa con NSGA-II
 4. Preserva la morfología urbana mientras mejora la distribución
 
Requisitos:
    pip install osmnx==1.9.3 networkx==3.3 geopandas shapely rtree numpy pandas tqdm folium pymoo==0.6.1.1

Uso:
    python ciudad_15min_reordenamiento.py --place "San Juan de Miraflores, Lima, Peru" --minutes 15 --iterations 5 --plot
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import os
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
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
    # Suprimir advertencia sobre módulos compilados
    try:
        from pymoo.config import Config
        Config.warnings['not_compiled'] = False
    except:
        pass
    PYMOO_OK = True
except Exception:
    PYMOO_OK = False

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
) -> Tuple[float, np.ndarray]:
    """
    Calcula la cobertura de accesibilidad
    Retorna: (cobertura_porcentaje, array_booleano_de_alcanzabilidad)
    """
    if services.empty or homes.empty:
        return 0.0, np.zeros(len(homes), dtype=bool)
    
    home_nodes = nearest_node_series(G, homes)
    serv_nodes = nearest_node_series(G, services)
    uniq_serv_nodes = list(set(serv_nodes.dropna().tolist()))
    
    if not uniq_serv_nodes:
        return 0.0, np.zeros(len(homes), dtype=bool)
    
    # Dijkstra multi-source para eficiencia
    lengths = nx.multi_source_dijkstra_path_length(G, uniq_serv_nodes, weight="travel_time")
    
    reachable = np.zeros(len(homes), dtype=bool)
    for i, (idx, hn) in enumerate(home_nodes.items()):
        t = lengths.get(hn, np.inf)
        reachable[i] = (t / 60.0) <= threshold_min
    
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
    """Inicialización para todas las categorías: 0=hogar, 1=health, 2=education, 3=greens, 4=work"""
    
    def __init__(self, n_homes: int, n_health: int, n_education: int, n_greens: int, n_work: int):
        super().__init__()
        self.n_homes = n_homes
        self.n_health = n_health
        self.n_education = n_education
        self.n_greens = n_greens
        self.n_work = n_work
    
    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        X = np.zeros((n_samples, n_var), dtype=int)
        
        rng = np.random.default_rng()
        total_assigned = self.n_homes + self.n_health + self.n_education + self.n_greens + self.n_work
        
        # Verificar que no excedamos el número de variables
        if total_assigned > n_var:
            raise ValueError(f"Total asignado ({total_assigned}) excede número de variables ({n_var})")
        
        for i in range(n_samples):
            x = np.zeros(n_var, dtype=int)
            # Asignar tipos de forma aleatoria pero respetando las cantidades
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
    
    def __init__(self, n_homes: int, n_health: int, n_education: int, n_greens: int, n_work: int, prob=0.2):
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
        
        # f1: Minimizar (1 - cobertura_health)
        if not health_locs.empty and not homes_locs.empty:
            cov_health, _ = calculate_coverage(self.G, homes_locs, health_locs, self.minutes)
            objectives.append(1.0 - cov_health)
        else:
            objectives.append(1.0)
        
        # f2: Minimizar (1 - cobertura_education)
        if not education_locs.empty and not homes_locs.empty:
            cov_education, _ = calculate_coverage(self.G, homes_locs, education_locs, self.minutes)
            objectives.append(1.0 - cov_education)
        else:
            objectives.append(1.0)
        
        # f3: Minimizar (1 - cobertura_greens)
        if not greens_locs.empty and not homes_locs.empty:
            cov_greens, _ = calculate_coverage(self.G, homes_locs, greens_locs, self.minutes)
            objectives.append(1.0 - cov_greens)
        else:
            objectives.append(1.0)
        
        # f4: Minimizar (1 - cobertura_work)
        if not work_locs.empty and not homes_locs.empty:
            cov_work, _ = calculate_coverage(self.G, homes_locs, work_locs, self.minutes)
            objectives.append(1.0 - cov_work)
        else:
            objectives.append(1.0)
        
        # f5: Minimizar número de cambios respecto a la configuración inicial
        # Contar cuántas ubicaciones cambiaron de tipo
        n_changes = int((x != self.initial_config).sum())
        # Normalizar por número total de ubicaciones (para que esté entre 0 y 1)
        change_ratio = n_changes / len(x) if len(x) > 0 else 1.0
        objectives.append(change_ratio)
        
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


def run_reordering_optimization(
    G: nx.MultiDiGraph,
    homes: gpd.GeoDataFrame,
    services: Dict[str, gpd.GeoDataFrame],
    target_category: str,
    minutes: float = 15.0,
    max_gen: int = 100,
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
    max_gen: int = 100,
    pop_size: int = 50
) -> Tuple[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame], pd.DataFrame, Dict[str, float]]:
    """
    Ejecuta optimización con reordenamiento para TODAS las categorías simultáneamente
    
    Returns:
        (nuevos_hogares, nuevos_servicios_por_categoria, frente_pareto, mejores_coberturas)
    """
    if not PYMOO_OK:
        raise RuntimeError("pymoo no está instalado")
    
    print(f"\n[NSGA-II] Iniciando optimización con reordenamiento para TODAS las categorías")
    print(f"  Generaciones: {max_gen}, Población: {pop_size}")
    
    problem = ReorderingProblemAllCategories(
        G, homes, services, minutes
    )
    
    # Usar inicialización factible, operadores personalizados y reparador para todas las categorías
    sampling = FeasibleSamplingAllCategories(
        n_homes=problem.n_homes,
        n_health=problem.n_health,
        n_education=problem.n_education,
        n_greens=problem.n_greens,
        n_work=problem.n_work
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
        prob=0.2
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
        return new_homes, new_services, pareto, best_covs
    
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
    pareto = pd.DataFrame({
        "1-cov_health": F_feas[:, 0],
        "1-cov_education": F_feas[:, 1],
        "1-cov_greens": F_feas[:, 2],
        "1-cov_work": F_feas[:, 3],
        "change_ratio": F_feas[:, 4]  # Proporción de cambios
    })
    pareto["solution_index"] = np.arange(len(pareto))
    
    # Elegir mejor solución (balance entre cobertura y minimización de cambios)
    # Normalizar objetivos (0-1) y balancear con peso en cambios
    norm = (pareto.iloc[:, :4] - pareto.iloc[:, :4].min()) / (pareto.iloc[:, :4].max() - pareto.iloc[:, :4].min() + 1e-9)
    norm_changes = pareto["change_ratio"] / (pareto["change_ratio"].max() + 1e-9)
    # Minimizar suma de coberturas (menor es mejor) y cambios (menor es mejor)
    # Peso 0.3 para cambios: preferir soluciones con menos cambios
    pareto["score"] = norm.sum(axis=1) + 0.3 * norm_changes
    
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
    
    return new_homes, new_services, pareto, best_covs


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
    max_gen: int = 100,
    pop_size: int = 50
) -> Tuple[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame], List[Dict]]:
    """
    Ejecuta optimización de TODAS las categorías simultáneamente (una sola iteración)
    
    Returns:
        (hogares_finales, servicios_finales_por_categoria, historial_metricas)
    """
    # Forzar a 1 iteración (todas las categorías juntas)
    n_iterations = 1
    
    print("\n" + "="*70)
    print("OPTIMIZACIÓN CON REORDENAMIENTO - TODAS LAS CATEGORÍAS JUNTAS")
    print("="*70)
    
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
    
    final_homes, final_services, pareto, best_covs = run_reordering_optimization_all_categories(
        G=G,
        homes=initial_homes,
        services=initial_services,
        minutes=minutes,
        max_gen=max_gen,
        pop_size=pop_size
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
    
    return final_homes, final_services, history


# -----------------------------
# VISUALIZACIÓN
# -----------------------------

try:
    import folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False


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
    parser.add_argument("--generations", type=int, default=100,
                       help="Generaciones por optimización NSGA-II")
    parser.add_argument("--population", type=int, default=50,
                       help="Tamaño de población NSGA-II")
    parser.add_argument("--categories", type=str, nargs='+',
                       default=["health", "education", "greens"],
                       help="Categorías a optimizar")
    parser.add_argument("--plot", action="store_true",
                       help="Generar mapa interactivo")
    parser.add_argument("--output-dir", type=str, default="outputs_reordenamiento",
                       help="Directorio de salida")
    
    args = parser.parse_args()
    
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
    
    # 3. OPTIMIZACIÓN ITERATIVA
    final_homes, final_services, history = iterative_reordering(
        G=G,
        initial_homes=homes,
        initial_services=services,
        categories=args.categories,
        minutes=args.minutes,
        n_iterations=args.iterations,
        max_gen=args.generations,
        pop_size=args.population
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
    
    # 6. GENERAR MAPA
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
