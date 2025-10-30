# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from PIL import Image
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD, value

# ===========================
#   Layout y Encabezado
# ===========================
st.set_page_config(page_title="Optimizador Producción Soldadura", layout="wide")

st.markdown(
    """
    <div style='text-align:center; padding:1rem; background-color:#f1f3f6; border-radius:10px; border:1px solid #ccc'>
        <h2 style='font-family:Arial; color:#1c1c1c;'>Optimizador Programación de Producción Soldadura</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Logo (opcional)
try:
    logo = Image.open("YAMAHA.PNG")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        st.image(logo, width=250)
except Exception:
    pass

# ===========================
#   Parámetros generales
# ===========================
st.header("📊 Parámetros de la planta")
lineas_disponibles = st.number_input(
    "Número máximo de líneas simultáneas activas por turno",
    min_value=1, max_value=8, value=3,
    help="Límite de líneas que pueden operar al mismo tiempo en un turno."
)
base_capacidad = st.number_input(
    "Minutos disponibles por turno (fallback si faltan en archivo de demanda)",
    min_value=60, max_value=1000, value=390, step=10
)

ALMACEN_MAX = 350.0  # Capacidad máxima de almacenamiento (total por turno)

# ===========================
#   Uploader Demanda
# ===========================
st.header("📂  Cargar archivo de demanda")
st.markdown("""
El archivo de demanda **debe** traer estas columnas mínimas:
- `Dia` (entero), `Turno` (1..N), `Modelo`, `Cantidad_Optima`,
- `Personas` (por turno), `Tiempo` (minutos disponibles por turno).
""")
archivo = st.file_uploader(
    "Carga tu archivo Excel con la programación de producción (demanda)",
    type=["xlsx"],
)

# Guard clause si no hay archivo
if archivo is None:
    st.info("Sube el Excel de demanda para continuar. Abajo aparecerán más controles.")
    st.stop()

# ===========================
#   Lectura y normalización Demanda
# ===========================
df = pd.read_excel(archivo)

def parse_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(',', '.')
    try:
        return float(s)
    except:
        return np.nan

# Columnas requeridas en demanda
required_cols = {"Modelo", "Dia", "Turno", "Cantidad_Optima", "Personas", "Tiempo"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Faltan columnas requeridas en el Excel de demanda: {sorted(missing)}")
    st.stop()

st.subheader("📄 Vista previa — Demanda")
st.dataframe(df.head(20), use_container_width=True)

# Mapeo Ensamble → Soldadura (normalización de modelos)
ensamble_a_soldadura = {
    "BUB1": "GPD155-A (BUB1)",
    "BF24": "T115FI (BF24)",
    "2MD8": "XTZ125 (2MD8)",
    "BGK3": "FZN150-A (BGK3)",
    "BGK2": "FZN150-A (BGK2)",
    "D241": "T115FL (D241)",
    "XTZ250": "B9L3/B9L4",  # etiqueta normalizada para XTZ250
}

def normaliza_modelo(raw: str) -> str:
    s = str(raw)
    # Si ya contiene nombre objetivo:
    for v in ensamble_a_soldadura.values():
        if v in s:
            return v
    # Si contiene un código clave:
    for k, v in ensamble_a_soldadura.items():
        if k in s:
            return v
    # Si trae separador '_' (p.ej. '1_BF24'): usa el último token para mapear
    if "_" in s:
        tail = s.split("_")[-1]
        for k, v in ensamble_a_soldadura.items():
            if k == tail:
                return v
    return s.strip()

# Tipos y normalizaciones
df["Dia"] = pd.to_numeric(df["Dia"], errors="coerce")

def _to_turno(z):
    if pd.isna(z):
        return None
    m = re.search(r"(\d+)", str(z))
    return int(m.group(1)) if m else None

df["Turno"] = df["Turno"].apply(_to_turno)

df["Cantidad_Optima"] = df["Cantidad_Optima"].apply(parse_num)
df["Personas"] = df["Personas"].apply(parse_num)
df["Tiempo"] = df["Tiempo"].apply(parse_num)

df["Modelo"] = df["Modelo"].astype(str).apply(normaliza_modelo)
modelos_validos = set(ensamble_a_soldadura.values())
df = df[df["Modelo"].isin(modelos_validos)].copy()

# Modelos presentes
modelos = sorted(df["Modelo"].dropna().unique().tolist())
if len(modelos) == 0:
    st.error("Tras filtrar, no quedaron modelos válidos. Revisa 'Modelo' y el mapeo.")
    st.stop()

# ===========================
#   Parámetros base y líneas permitidas (fallback)
# ===========================
PitchTime_base = {
    "GPD155-A (BUB1)": 3.65,
    "T115FI (BF24)": 3.75,
    "XTZ125 (2MD8)": 3.75,
    "XTZ125 (2MD7)": 3.75,
    "FZN150-A (BGK3)": 3.0,
    "FZN150-A (BGK2)": 3.0,
    "B9L3/B9L4": 3.75,  # XTZ250
}
LeadTime_base = {
    "GPD155-A (BUB1)": 28.0,
    "T115FI (BF24)": 18.75,
    "XTZ125 (2MD8)": 15.0,
    "XTZ125 (2MD7)": 15.0,
    "FZN150-A (BGK3)": 15.0,
    "FZN150-A (BGK2)": 15.0,
    "B9L3/B9L4": 15.0,
}

# Líneas permitidas por modelo (fallback si no hay curvas con línea)
linea_permitida = {
    "GPD155-A (BUB1)": [1],
    "XTZ125 (2MD7)": [2],
    "XTZ125 (2MD8)": [2],
    "FZN150-A (BGK2)": [3],
    "FZN150-A (BGK3)": [3],
    "T115FI (BF24)": [2, 3],
    "B9L3/B9L4": [4],  # XTZ250 en línea 4
}

missing_models = [m for m in modelos if m not in linea_permitida]
if missing_models:
    st.error(f"Faltan reglas en 'linea_permitida' para: {missing_models}")
    st.stop()
empty_rules = [m for m in modelos if len(linea_permitida.get(m, [])) == 0]
if empty_rules:
    st.error(f"'linea_permitida' tiene listas vacías para: {empty_rules}")
    st.stop()

# Ajustes opcionales Pitch/Lead (fallback cuando no hay curvas)
st.subheader("⚙️ Ajustes de Pitch/Lead (fallback, si no hay curvas)")
pitchtime_input, leadtime_input = {}, {}
for modelo in modelos:
    c1, c2 = st.columns(2)
    with c1:
        pitch = st.number_input(
            f"Pitch Time para {modelo} (min/unid) [fallback]",
            min_value=0.1, value=float(PitchTime_base.get(modelo, 4.0)), step=0.05, key=f"pt_{modelo}",
        )
    with c2:
        lead = st.number_input(
            f"Lead Time para {modelo} (min, setup)",
            min_value=0.0, value=float(LeadTime_base.get(modelo, 15.0)), step=0.5, key=f"lt_{modelo}",
        )
    pitchtime_input[modelo] = float(pitch)
    leadtime_input[modelo] = float(lead)

# ===========================
#   Curvas de capacidad (opcional)
#   PT = Tiempo_Produccion / Capacidad (o Tiempo_necesario/Capacidad)
# ===========================
st.header("📈 Curvas de capacidad (opcional)")
st.markdown("""
Archivo con columnas (nombres flexibles):
- `Modelo`, `Linea` / `Línea`, `Personas`, `Capacidad` (u), `Tiempo_Produccion` **o** `Tiempo_necesario` (min),
- (opcional) `Horas`, `Vigente`.
Se calculará **PT = Tiempo / Capacidad** (min/unidad) por **Modelo–Línea–Personas**.
""")
curvas_file = st.file_uploader(
    "Cargar curvas de capacidad (CSV/Excel)",
    type=["csv", "xlsx", "xls"],
    key="curvas_capacidad"
)

def _read_any(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file)
    return pd.read_csv(file)

def _norm_cols(df_in):
    df2 = df_in.copy()
    df2.columns = [c.strip().lower() for c in df2.columns]
    return df2

# Confirmado: BF2 → T115FI (BF24)
alias_a_modelo = {
    "BF2": "T115FI (BF24)",     # confirmado
    "BF24": "T115FI (BF24)",
    "BUB1": "GPD155-A (BUB1)",
    "2MD8": "XTZ125 (2MD8)",
    "2MD7": "XTZ125 (2MD7)",
    "BGK3": "FZN150-A (BGK3)",
    "BGK2": "FZN150-A (BGK2)",
    "B9L3": "B9L3/B9L4",
    "B9L4": "B9L3/B9L4",
    "XTZ250": "B9L3/B9L4",
}

def normaliza_modelo_curva(s: str) -> str:
    s = str(s).strip()
    if s in alias_a_modelo.values():
        return s
    for k, v in alias_a_modelo.items():
        if k in s:
            return v
    return s

def parse_bool_vigente(x) -> bool:
    s = str(x).strip().lower()
    return s in {"si", "sí", "true", "1", "vigente", "y", "yes"}

pt_lookup = None
if curvas_file:
    try:
        raw = _read_any(curvas_file)
        c = _norm_cols(raw)

        # Renombrar columnas alternativas
        ren = {}
        if 'línea' in c.columns: ren['línea'] = 'linea'
        if 'tiempo_produccion' not in c.columns and 'tiempo_necesario' in c.columns:
            ren['tiempo_necesario'] = 'tiempo_produccion'
        c = c.rename(columns=ren)

        required = {"modelo", "capacidad", "linea", "personas"}
        missing = required - set(c.columns)
        if missing:
            raise ValueError(f"Faltan columnas mínimas en curvas: {missing}")
        if 'tiempo_produccion' not in c.columns:
            raise ValueError("Falta 'Tiempo_Produccion' (o 'Tiempo_necesario') en el archivo de curvas.")

        # Filtrar vigentes si aplica
        if "vigente" in c.columns:
            c['vigente_bool'] = c['vigente'].apply(parse_bool_vigente)
            if c['vigente_bool'].any():
                c = c[c['vigente_bool']].copy()
            else:
                st.warning("La columna 'Vigente' no marcó filas vigentes. Se usarán todas.")

        # Limpieza numérica
        for col in ["capacidad", "tiempo_produccion", "personas", "linea", "horas"]:
            if col in c.columns:
                c[col] = pd.to_numeric(c[col], errors='coerce')

        # Filas válidas
        c = c[
            (c['capacidad'] > 0) &
            (c['tiempo_produccion'] > 0) &
            c['personas'].notna() &
            c['linea'].notna()
        ].copy()

        # Normalizar modelo y calcular PT
        c['modelo_norm'] = c['modelo'].astype(str).apply(normaliza_modelo_curva)
        c['pt_min_unid'] = c['tiempo_produccion'] / c['capacidad']  # PT = Tiempo / Capacidad

        # Lookup por (modelo, linea, personas) - usar mediana si hay varias filas
        pt_lookup = (
            c.groupby(['modelo_norm', 'linea', 'personas'], as_index=False)['pt_min_unid']
             .median()
             .rename(columns={'modelo_norm': 'modelo', 'pt_min_unid': 'pt'})
        )

        st.success("Curvas de capacidad cargadas ✓ (PT = Tiempo/Capacidad)")
        with st.expander("👀 PT efectivo por Modelo–Línea–Personas (vista previa)"):
            st.dataframe(pt_lookup.head(30), use_container_width=True)

        st.session_state['pt_lookup'] = pt_lookup

        # Derivar líneas habilitadas por modelo desde curvas
        try:
            lineas_desde_curvas = (
                pt_lookup.groupby('modelo')['linea']
                         .apply(lambda s: sorted(pd.unique(s.dropna().astype(int))))
                         .to_dict()
            )
        except Exception:
            lineas_desde_curvas = {}
        st.session_state['lineas_desde_curvas'] = lineas_desde_curvas

    except Exception as e:
        st.error(f"Error leyendo curvas de capacidad: {e}")

# Toggle para usar líneas desde curvas si existen
usar_lineas_de_curvas = st.toggle(
    "Restringir líneas a las definidas en el archivo de curvas si existen",
    value=True,
    help="Si está activo y el archivo de curvas indica Líneas por Modelo, se usarán esas líneas. "
         "Si no hay curvas o no indican línea, se usa 'linea_permitida' del código."
)
st.session_state['usar_lineas_de_curvas'] = usar_lineas_de_curvas

# ===========================
#   Construcción de mapas Tiempo/Personas por (Dia,Turno)
# ===========================
tiempo_por_dt = (df.groupby(["Dia", "Turno"])["Tiempo"].max().dropna().to_dict())
personas_por_dt = (df.groupby(["Dia", "Turno"])["Personas"].max().dropna().to_dict())

st.session_state["tiempo_por_dt"] = { (int(d), int(t)): float(v) for (d,t),v in tiempo_por_dt.items() }
st.session_state["personas_por_dt"] = { (int(d), int(t)): float(v) for (d,t),v in personas_por_dt.items() }

# ===========================
#   Debug básico
# ===========================
show_debug = st.checkbox("🛠 Mostrar depuración", value=False)
if show_debug:
    st.write("Modelos (normalizados):", modelos)
    st.write("Días únicos:", sorted(df["Dia"].dropna().unique().tolist()))
    st.write("Turnos únicos:", sorted(df["Turno"].dropna().unique().tolist()))

# ===========================
#   BOTÓN: Ejecutar Optimización
# ===========================
if st.button("🚀 Ejecutar optimización"):
    PitchTime = pitchtime_input
    LeadTime = leadtime_input

    # Rezago de fabricación (turnos)
    rezago = 3

    # Reindexar días 1..D
    dias_sorted = sorted([int(x) for x in df["Dia"].dropna().unique().tolist()])
    dia_to_idx = {d: i + 1 for i, d in enumerate(dias_sorted)}

    # Turnos por día
    turnos_unicos = sorted([int(x) for x in df["Turno"].dropna().unique().tolist()])
    if len(turnos_unicos) == 0:
        st.error("No se detectaron turnos válidos en el Excel de demanda.")
        st.stop()
    n_turnos_por_dia = int(max(turnos_unicos))

    # Turnos absolutos
    turnos = list(range(1, len(dias_sorted) * n_turnos_por_dia + rezago + 1))

    # Demanda por (modelo, turno abs)
    demanda_t = {(m, t): 0.0 for m in modelos for t in turnos}
    for _, r in df.iterrows():
        m = r["Modelo"]; d = r["Dia"]; tu = r["Turno"]; qty = r["Cantidad_Optima"]
        if pd.isna(m) or pd.isna(d) or pd.isna(tu) or pd.isna(qty):
            continue
        d_idx = dia_to_idx.get(int(d))
        t_abs = (d_idx - 1) * n_turnos_por_dia + int(tu)
        if t_abs in turnos and (m, t_abs) in demanda_t:
            demanda_t[(m, t_abs)] += float(qty)

    total_demand = sum(demanda_t.values())
    if show_debug:
        st.write("Demanda total mapeada:", total_demand)
    if total_demand == 0:
        st.warning("La demanda mapeada resultó 0. Revisa 'Dia', 'Turno' y 'Modelo'.")

    # ========= Líneas candidatas (curvas ∪ mapping) =========
    pt_lookup = st.session_state.get('pt_lookup', None)
    lines_from_curves = sorted(
        pt_lookup['linea'].dropna().astype(int).unique().tolist()
    ) if (pt_lookup is not None and 'linea' in pt_lookup.columns) else []
    lines_from_mapping = sorted({l for m in modelos for l in linea_permitida.get(m, [])})
    lineas_all = sorted(set(lines_from_curves) | set(lines_from_mapping))
    if not lineas_all:
        lineas_all = [1, 2, 3]  # fallback
    if show_debug:
        st.write("Líneas candidatas:", lineas_all)

    # ========= Capacidad (min) y Personas por (línea, turno absoluto) =========
    from collections import defaultdict
    cap_by_absL = defaultdict(float)   # (l, t_abs) -> minutos
    personas_by_absL = {}              # (l, t_abs) -> personas

    tiempo_por_dt = st.session_state.get("tiempo_por_dt", {})
    personas_por_dt = st.session_state.get("personas_por_dt", {})

    def split_abs_turn(t_abs: int):
        dia_idx = (t_abs - 1) // n_turnos_por_dia + 1
        turno_idx = (t_abs - 1) % n_turnos_por_dia + 1
        return int(dia_idx), int(turno_idx)

    for t_abs in turnos:
        d_idx, tu_idx = split_abs_turn(t_abs)
        minutos_turno = tiempo_por_dt.get((d_idx, tu_idx), None)
        personas_turno = personas_por_dt.get((d_idx, tu_idx), None)
        if minutos_turno is None or pd.isna(minutos_turno):
            minutos_turno = float(base_capacidad)
        for l in lineas_all:
            cap_by_absL[(l, t_abs)] = float(minutos_turno)
            if personas_turno is not None and not pd.isna(personas_turno):
                personas_by_absL[(l, t_abs)] = float(personas_turno)

    # (Opcional) Turnos especiales con -60 min
    turnos_menos_60 = {3, 4, 7, 8, 17, 18, 21, 22, 31, 32, 35, 36, 45, 46, 49, 50}
    for (l, t_abs), v in list(cap_by_absL.items()):
        if t_abs in turnos_menos_60:
            cap_by_absL[(l, t_abs)] = max(0.0, v - 60.0)

    def capacidad_linea_turno(l: int, t_abs: int) -> float:
        return float(cap_by_absL.get((int(l), int(t_abs)), 0.0))

    # ========= Líneas habilitadas por modelo (curvas → mapping) =========
    usar_lineas_de_curvas = st.session_state.get('usar_lineas_de_curvas', True)
    lineas_desde_curvas = {}
    if pt_lookup is not None and 'linea' in pt_lookup.columns:
        try:
            lineas_desde_curvas = (
                pt_lookup.groupby('modelo')['linea']
                         .apply(lambda s: sorted(pd.unique(s.dropna().astype(int))))
                         .to_dict()
            )
        except Exception:
            lineas_desde_curvas = {}

    def allowed_lines_for_model(m: str):
        if usar_lineas_de_curvas and (m in lineas_desde_curvas) and len(lineas_desde_curvas[m]) > 0:
            ls = [l for l in lineas_desde_curvas[m] if l in lineas_all]
        else:
            ls = [l for l in linea_permitida.get(m, []) if l in lineas_all]
        return sorted(ls)

    lineas_habilitadas = {m: allowed_lines_for_model(m) for m in modelos}
    models_without_lines = [m for m, ls in lineas_habilitadas.items() if not ls]
    if models_without_lines:
        st.error(f"Estos modelos no tienen líneas habilitadas presentes en capacidad/curvas: {models_without_lines}. "
                 "Revisa curvas o 'linea_permitida'.")
        st.stop()

    # ========= PT efectivo y Fuente (curvas o fallback) =========
    CURVES_INCLUDE_SETUP = True  # Si True: al usar PT de curvas NO sumar setup ni restarlo en capacity bound

    def get_pt_eff_with_source(modelo: str, linea: int, t_abs: int, default_pt: float):
        """Devuelve (pt, from_curve: bool)."""
        personas = personas_by_absL.get((int(linea), int(t_abs)), None)
        if pt_lookup is None or personas is None or np.isnan(personas):
            return float(default_pt), False  # PT viene de UI/código
        df_pt = pt_lookup
        sel = df_pt[(df_pt['modelo'] == modelo) & (df_pt['linea'] == int(linea))]
        if sel.empty:
            sel = df_pt[(df_pt['modelo'] == modelo)]
            if sel.empty:
                return float(default_pt), False
        idx = (sel['personas'] - personas).abs().idxmin()
        pt_val = float(sel.loc[idx, 'pt'])
        if not (pt_val > 0):
            return float(default_pt), False
        return pt_val, True  # PT proviene de curvas

    PT_eff, PT_from_curve = {}, {}
    for m in modelos:
        for l in lineas_all:
            for t in turnos:
                pt_val, is_curve = get_pt_eff_with_source(m, l, t, pitchtime_input[m])
                PT_eff[(m, l, t)] = pt_val
                PT_from_curve[(m, l, t)] = is_curve

    # ========= Modelo de optimización (PuLP) =========
    model = LpProblem("Produccion_Soldadura", LpMinimize)

    # Variables (producción entera)
    x = {(m, l, t): LpVariable(f"x_{m}_{l}_{t}", lowBound=0, cat="Integer")
         for m in modelos for l in lineas_all for t in turnos}
    y = {(m, l, t): LpVariable(f"y_{m}_{l}_{t}", lowBound=0, upBound=1, cat="Binary")
         for m in modelos for l in lineas_all for t in turnos}
    uso_linea = {(l, t): LpVariable(f"uso_linea_{l}_{t}", lowBound=0, upBound=1, cat="Binary")
                 for l in lineas_all for t in turnos}
    i = {(m, t): LpVariable(f"inv_{m}_{t}", lowBound=0, cat="Continuous")
         for m in modelos for t in turnos}
    i_ini = {(m, t): LpVariable(f"inv_ini_{m}_{t}", lowBound=0, cat="Continuous")
             for m in modelos for t in turnos if t <= rezago}

    # --- Objetivo: minimizar inventario inicial (t <= R) ---
    model += lpSum(i_ini[m, t] for m in modelos for t in turnos if t <= rezago)

    # A lo sumo una familia por turno global para cada modelo
    for t in turnos:
        for m in modelos:
            model += lpSum(y[m, l, t] for l in lineas_all) <= 1

    # Bound de producción por activación (usa PT y, si PT es de curvas y flag activo, NO resta setup)
    for m in modelos:
        for l in lineas_all:
            for t in turnos:
                if l not in lineas_habilitadas[m]:
                    model += y[m, l, t] == 0
                    model += x[m, l, t] == 0
                    continue
                pt = float(PT_eff[(m, l, t)])
                lt = 0.0 if (PT_from_curve[(m, l, t)] and CURVES_INCLUDE_SETUP) else float(leadtime_input[m])
                cap_lt = capacidad_linea_turno(l, t)
                max_prod = max(0.0, (cap_lt - lt) / pt) if pt > 0 else 0.0
                model += x[m, l, t] <= max_prod * y[m, l, t]

    # A lo sumo un modelo por línea y turno + enlace a uso_linea
    for l in lineas_all:
        for t in turnos:
            model += lpSum(y[m, l, t] for m in modelos) <= 1
            for m in modelos:
                model += y[m, l, t] <= uso_linea[l, t]

    # Límite de líneas simultáneas activas por turno (control UI)
    for t in turnos:
        model += lpSum(uso_linea[l, t] for l in lineas_all) <= int(lineas_disponibles)

    # Desactivar líneas sin minutos en un turno
    for l in lineas_all:
        for t in turnos:
            if capacidad_linea_turno(l, t) <= 1e-6:
                model += uso_linea[l, t] == 0

    # Restricción de capacidad por línea/turno
    for l in lineas_all:
        for t in turnos:
            model += (
                lpSum(
                    PT_eff[(m, l, t)] * x[m, l, t]
                    + (0.0 if (PT_from_curve[(m, l, t)] and CURVES_INCLUDE_SETUP) else leadtime_input[m]) * y[m, l, t]
                    for m in modelos
                )
                <= capacidad_linea_turno(l, t)
            ), f"Cap_L{l}_T{t}"

    # --- Balance de inventario con rezago (COBERTURA ESTRICTA en t <= R) ---
    for m in modelos:
        valid_lines = lineas_habilitadas[m]
        for t in turnos:
            if t >= rezago + 1:
                model += i[m, t] == i[m, t - 1] \
                                  + lpSum(x[m, l, t - rezago] for l in valid_lines) \
                                  - demanda_t[(m, t)]
            else:
                # En los primeros R turnos, la producción aún no llega:
                # el inventario inicial debe cubrir la demanda del turno.
                model += i[m, t] == i_ini[m, t] - demanda_t[(m, t)]

    # --- Límite de almacenamiento total por turno ---
    for t in turnos:
        model += lpSum(i[m, t] for m in modelos) <= ALMACEN_MAX, f"CapAlmacen_T{t}"

    # (Opcional) Limitar también el inventario inicial por turno en t <= R
    for t in turnos:
        if t <= rezago:
            model += lpSum(i_ini[m, t] for m in modelos if (m, t) in i_ini) <= ALMACEN_MAX, f"CapAlmacenInicial_T{t}"

    # Resolver (CBC con parámetros recomendados)
    solver = PULP_CBC_CMD(msg=False, threads=4, timeLimit=60, gapRel=0.001)
    model.solve(solver)

    # ---------------- Resultados ----------------
    status_str = LpStatus.get(model.status, str(model.status))
    st.write(f"🔎 Estado del solver: **{status_str}**")

    if status_str == "Optimal":
        resumen = []

        def split_abs_turn_local(t_abs: int):
            dia_idx = (t_abs - 1) // n_turnos_por_dia + 1
            turno_idx = (t_abs - 1) % n_turnos_por_dia + 1
            return int(dia_idx), int(turno_idx)

        for t in turnos:
            d_idx, tu_idx = split_abs_turn_local(t)
            for l in lineas_all:
                cap = capacidad_linea_turno(l, t)
                pers = personas_by_absL.get((l, t), None)
                used = 0.0
                alguna_fila = False
                for m in modelos:
                    x_val = value(x[m, l, t]) or 0.0
                    y_val = value(y[m, l, t]) or 0.0
                    if x_val > 0 or y_val > 0:
                        pt_used = PT_eff[(m, l, t)]
                        # Setup = 0 si PT proviene de curvas y flag activo
                        setup_minutes = 0.0 if (PT_from_curve[(m, l, t)] and CURVES_INCLUDE_SETUP) else leadtime_input[m] * y_val
                        prod = x_val * pt_used
                        total = prod + setup_minutes
                        used += total
                        alguna_fila = True
                        resumen.append({
                            "Dia": d_idx,
                            "Turno": tu_idx,
                            "TurnoAbs": t,
                            "Línea": l,
                            "Modelo": m if y_val > 0 else "-",
                            "Cantidad a producir": round(x_val, 2),
                            "PT usado (min/u)": round(pt_used, 3),
                            "Tiempo producción (min)": round(prod, 2),
                            "Tiempo setup (min)": round(setup_minutes, 2),
                            "Tiempo total (min)": round(total, 2),
                            "Capacidad disponible (min)": round(cap, 2),
                            "Personas turno": (None if pers is None else int(pers)),
                            "% Uso": f"{round((total / cap) * 100, 1)}%" if cap > 0 else "N/A",
                        })
                # Si la línea no se usó, fila vacía
                if not alguna_fila:
                    resumen.append({
                        "Dia": d_idx,
                        "Turno": tu_idx,
                        "TurnoAbs": t,
                        "Línea": l,
                        "Modelo": "-",
                        "Cantidad a producir": 0,
                        "PT usado (min/u)": None,
                        "Tiempo producción (min)": 0,
                        "Tiempo setup (min)": 0,
                        "Tiempo total (min)": 0,
                        "Capacidad disponible (min)": round(cap, 2),
                        "Personas turno": (None if pers is None else int(pers)),
                        "% Uso": "0%" if cap > 0 else "N/A",
                    })

        df_uso_lineas = pd.DataFrame(resumen)

        st.success("✅ Optimización completada")
        st.subheader("📋 Resumen de uso de líneas")
        st.dataframe(df_uso_lineas, use_container_width=True)

        # ================= Inventarios: inicial (i_ini) y por turno (i) =================
        st.subheader("📦 Inventario inicial (i_ini) y trayectoria de inventario (i)")

        # Inventario inicial (solo t <= rezago)
        ini_rows = []
        for m in modelos:
            for t in turnos:
                if t <= rezago:
                    inv0 = value(i_ini[m, t]) if (m, t) in i_ini else 0.0
                    d_idx, tu_idx = split_abs_turn_local(t)
                    ini_rows.append({
                        "Modelo": m,
                        "Dia": d_idx,
                        "Turno": tu_idx,
                        "TurnoAbs": t,
                        "Inventario_Inicial (i_ini)": round(float(inv0), 2),
                    })

        df_i_ini = pd.DataFrame(ini_rows).sort_values(["Modelo", "TurnoAbs"])
        st.markdown("**Inventario inicial por modelo en los primeros `rezago` turnos:**")
        st.dataframe(df_i_ini, use_container_width=True)

        # Inventario por turno (todo el horizonte)
        inv_rows = []
        for m in modelos:
            for t in turnos:
                inv = value(i[m, t]) or 0.0
                d_idx, tu_idx = split_abs_turn_local(t)
                inv_rows.append({
                    "Modelo": m,
                    "Dia": d_idx,
                    "Turno": tu_idx,
                    "TurnoAbs": t,
                    "Inventario (i)": round(float(inv), 2),
                })

        df_inv = pd.DataFrame(inv_rows).sort_values(["Modelo", "TurnoAbs"])
        st.markdown("**Inventario por modelo y turno (trayectoria completa):**")
        st.dataframe(df_inv, use_container_width=True)

        # Resumen final por modelo (inventario en el último turno)
        last_t = max(turnos)
        df_inv_fin = (df_inv[df_inv["TurnoAbs"] == last_t]
                      .rename(columns={"Inventario (i)": "Inventario_Final"})
                      [["Modelo", "Inventario_Final"]]
                      .sort_values("Modelo"))
        st.markdown("**Inventario final por modelo (último turno del horizonte):**")
        st.dataframe(df_inv_fin, use_container_width=True)

        # Diagnóstico de almacenamiento
        inv_total_por_turno = []
        for t in turnos:
            total_t = sum((value(i[m, t]) or 0.0) for m in modelos)
            inv_total_por_turno.append({"TurnoAbs": t, "Inv_Total": total_t})
        df_inv_total = pd.DataFrame(inv_total_por_turno)
        peak = df_inv_total["Inv_Total"].max()
        t_peak = df_inv_total.loc[df_inv_total["Inv_Total"].idxmax(), "TurnoAbs"]
        st.subheader("🏭 Diagnóstico de almacenamiento")
        st.metric("Inventario total pico", f"{peak:,.2f} u", help=f"TurnoAbs con pico: {int(t_peak)}")
        if peak >= ALMACEN_MAX - 1e-6:
            st.warning(f"El límite de almacenamiento ({int(ALMACEN_MAX)} u) estuvo activo en al menos un turno.")
        else:
            st.info("No se alcanzó el límite de almacenamiento en el horizonte.")

        # Descarga en Excel (plan + inventarios)
        buf2 = BytesIO()
        with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
            df_uso_lineas.to_excel(writer, index=False, sheet_name="Plan")
            df_i_ini.to_excel(writer, index=False, sheet_name="InventarioInicial")
            df_inv.to_excel(writer, index=False, sheet_name="InventarioPorTurno")
            df_inv_fin.to_excel(writer, index=False, sheet_name="InventarioFinal")

        st.download_button(
            "⬇️ Descargar inventarios (Excel)",
            data=buf2.getvalue(),
            file_name="inventarios_soldadura.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Visualización rápida
        with st.expander("📊 Ver gráficos de inventario por modelo"):
            try:
                import altair as alt
                chart = alt.Chart(df_inv).mark_line(point=True).encode(
                    x=alt.X("TurnoAbs:Q", title="Turno absoluto"),
                    y=alt.Y("Inventario (i):Q", title="Inventario (unidades)"),
                    color=alt.Color("Modelo:N", title="Modelo"),
                    tooltip=["Modelo", "Dia", "Turno", "TurnoAbs", "Inventario (i)"]
                ).properties(height=300, width=800)
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                # Fallback simple con line_chart
                try:
                    pivot = df_inv.pivot(index="TurnoAbs", columns="Modelo", values="Inventario (i)")
                    st.line_chart(pivot, height=300)
                except Exception:
                    st.info("No se pudo renderizar el gráfico de inventarios. Revisa los datos.")
    else:
        st.error("El solver no encontró solución óptima. Revisa restricciones, capacidad y demanda.")