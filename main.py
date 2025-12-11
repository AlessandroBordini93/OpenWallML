# main.py
import io
import contextlib
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import openseespy.opensees as ops

# ============================================================
#  COSTANTI GLOBALI
# ============================================================

L = 4.0
H = 6.0

# Mesh più leggera per velocizzare
NX_NL = 30
NY_NL = 60

TARGET_DISP_MM = 15.0

CORDOLI_Y = [
    (2.7, 3.0),
    (5.7, 6.0),
]

MARGIN = 0.30
PIER_MIN = 0.30  # maschio orizzontale minimo


# ============================================================
#  FUNZIONI GEOMETRIA
# ============================================================

def inside_opening(x: float, y: float, openings: List[Tuple[float, float, float, float]]) -> bool:
    """True se il punto (x,y) è dentro una delle aperture (solo interno, NON il bordo)."""
    for (x1, x2, y1, y2) in openings:
        if (x > x1) and (x < x2) and (y > y1) and (y < y2):
            return True
    return False


def openings_valid(openings, cordoli_y, margin) -> bool:
    """
    Controlla che:
      - le aperture siano tutte dentro la parete con margin
      - non siano troppo vicine ai cordoli (>= margin)
      - non si sovrappongano e non siano più vicine di 'margin' tra loro
      - i maschi orizzontali non siano più piccoli di PIER_MIN
    """
    # 1) limiti della parete + maschio ai bordi
    for (x1, x2, y1, y2) in openings:
        # bordo parete: voglio almeno PIER_MIN di muratura
        if x1 < PIER_MIN or x2 > (L - PIER_MIN):
            return False

        if not (0.0 + margin <= x1 < x2 <= L - margin):
            return False
        if not (0.0 + margin <= y1 < y2 <= H - margin):
            return False

        # 2) distanza da cordoli
        for (yc1, yc2) in cordoli_y:
            if not (y2 <= yc1 - margin or y1 >= yc2 + margin):
                return False

    # 3) distanza tra aperture
    n = len(openings)
    for i in range(n):
        x1i, x2i, y1i, y2i = openings[i]
        for j in range(i + 1, n):
            x1j, x2j, y1j, y2j = openings[j]

            # gap in x e in y
            dx_gap = max(0.0, max(x1i, x1j) - min(x2i, x2j))
            dy_gap = max(0.0, max(y1i, y1j) - min(y2i, y2j))

            # sovrapposizione in quota?
            overlap_y = not (y2i <= y1j or y2j <= y1i)

            # se stanno sullo stesso piano (overlap in y), voglio un maschio >= PIER_MIN
            if overlap_y and dx_gap < PIER_MIN:
                return False

            # se sono "diagonali", tengo comunque un minimo di separazione generica = margin
            if dx_gap < margin and dy_gap < margin:
                return False

    return True


def K_from_E_nu(E: float, nu: float) -> float:
    return E / (3.0 * (1.0 - 2.0 * nu))


def G_from_E_nu(E: float, nu: float) -> float:
    return E / (2.0 * (1.0 + nu))


# ============================================================
#  MODELLO NON LINEARE J2
# ============================================================

def build_wall_J2(openings, nx: int, ny: int):
    """
    Modello NON lineare J2:
      - parete 4 x 6 m
      - quad4
      - aperture (input)
      - cordoli (materiale più rigido)
      - J2Plasticity -> PlaneStress per muratura e cordoli
    """
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 2)

    dx = L / nx
    dy = H / ny

    # -------------------------
    # NODI (NO dentro le aperture)
    # -------------------------
    node_tags: Dict[Tuple[int, int], int] = {}
    tag = 1

    for j in range(ny + 1):
        y = j * dy
        for i in range(nx + 1):
            x = i * dx

            if inside_opening(x, y, openings):
                continue

            ops.node(tag, x, y)
            node_tags[(i, j)] = tag
            tag += 1

    # Vincoli alla base
    for i in range(nx + 1):
        key = (i, 0)
        if key in node_tags:
            ops.fix(node_tags[key], 1, 1)

    # -------------------------
    # MATERIALI NON LINEARI – J2Plasticity -> PlaneStress
    # -------------------------
    # Muratura (valori indicativi)
    E_mur   = 1.5e9
    nu_mur  = 0.15
    sig0_m  = 0.5e6
    sigInf_m = 2.0e6
    delta_m  = 8.0
    H_m      = 0.0

    # Calcestruzzo cordoli (più rigido)
    E_cord   = 30e9
    nu_cord  = 0.20
    sig0_c   = 6.0e6
    sigInf_c = 25.0e6
    delta_c  = 6.0
    H_c      = 0.0

    K_m = K_from_E_nu(E_mur, nu_mur)
    G_m = G_from_E_nu(E_mur, nu_mur)
    K_c = K_from_E_nu(E_cord, nu_cord)
    G_c = G_from_E_nu(E_cord, nu_cord)

    # 3D J2Plasticity per muratura e c.a.
    matTag_mur_3D  = 10
    matTag_cord_3D = 20

    ops.nDMaterial('J2Plasticity', matTag_mur_3D,
                   K_m, G_m, sig0_m, sigInf_m, delta_m, H_m)
    ops.nDMaterial('J2Plasticity', matTag_cord_3D,
                   K_c, G_c, sig0_c, sigInf_c, delta_c, H_c)

    # Wrapper PlaneStress
    matTag_mur   = 1
    matTag_cord  = 2
    ops.nDMaterial('PlaneStress', matTag_mur,  matTag_mur_3D)
    ops.nDMaterial('PlaneStress', matTag_cord, matTag_cord_3D)

    t = 0.25  # spessore [m]

    # -------------------------
    # ELEMENTI QUAD4
    # -------------------------
    eleTag = 1
    for j in range(ny):
        for i in range(nx):
            keys = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]

            if not all(k in node_tags for k in keys):
                continue

            yc = (j + 0.5) * dy

            this_mat = matTag_mur
            for (y1c, y2c) in CORDOLI_Y:
                if (yc >= y1c) and (yc <= y2c):
                    this_mat = matTag_cord
                    break

            n1 = node_tags[keys[0]]
            n2 = node_tags[keys[1]]
            n3 = node_tags[keys[2]]
            n4 = node_tags[keys[3]]

            ops.element(
                'quad', eleTag, n1, n2, n3, n4,
                t, 'PlaneStress', this_mat, 0.0, 0.0, 0.0
            )
            eleTag += 1

    # -------------------------
    # CARICO ORIZZONTALE (PUSHOVER)
    # -------------------------
    top_nodes = [node_tags[(i, ny)] for i in range(nx + 1)
                 if (i, ny) in node_tags]

    if not top_nodes:
        raise RuntimeError("Nessun nodo in sommità: layout aperture troppo aggressivo.")

    # nodo di controllo centrale tra quelli in sommità
    control_node = top_nodes[len(top_nodes) // 2]

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    Ptot = 100e3      # [N]
    Pnode = Ptot / len(top_nodes)

    for nd in top_nodes:
        ops.load(nd, Pnode, 0.0)

    return node_tags, control_node


def shear_at_target_disp(disp_mm: np.ndarray,
                         shear_kN: np.ndarray,
                         target_mm: float = TARGET_DISP_MM) -> Union[float, None]:
    """
    Interpola il taglio alla base alla deformata target_mm.
    Restituisce None se la curva non arriva a target_mm.
    """
    if len(disp_mm) < 2:
        return None
    if float(np.max(disp_mm)) < target_mm:
        return None

    return float(np.interp(target_mm, disp_mm, shear_kN))


# ============================================================
#  ANALISI PUSHOVER (SOLO CURVA)
# ============================================================

def run_pushover_nonlinear(openings,
                           nx: int = NX_NL,
                           ny: int = NY_NL,
                           target_mm: float = TARGET_DISP_MM,
                           max_steps: int = 100,
                           dU: float = 0.0002,
                           verbose: bool = False) -> Dict[str, Any]:
    """
    Esegue pushover non lineare J2.
    Restituisce:
      {
        "status": "ok"/"error",
        "message": str o None,
        "disp_mm": [...],
        "shear_kN": [...],
        "V_target": float o None
      }
    """
    try:
        if not openings_valid(openings, CORDOLI_Y, MARGIN):
            return {
                "status": "error",
                "message": "openings_invalid",
                "disp_mm": [],
                "shear_kN": [],
                "V_target": None
            }

        node_tags, control_node = build_wall_J2(openings, nx, ny)

        ops.constraints('Plain')
        ops.numberer('RCM')
        ops.system('BandGeneral')

        ops.test('NormUnbalance', 1.0e-4, 15)
        ops.algorithm('Newton')

        ops.integrator('DisplacementControl', control_node, 1, dU)
        ops.analysis('Static')

        disp_mm: List[float] = []
        shear_kN: List[float] = []

        j2_problem = False
        lin_problem = False

        buf = io.StringIO()

        for step in range(max_steps):
            buf.truncate(0)
            buf.seek(0)

            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ok = ops.analyze(1)

            log_text = buf.getvalue()

            if ("J2-plasticity" in log_text) and ("More than 25 iterations" in log_text):
                j2_problem = True
                if verbose:
                    print(f"  [NL] Problema J2-plasticity allo step {step}, analisi interrotta.")
                break

            if ("factorization failed" in log_text) or ("matrix singular" in log_text):
                lin_problem = True
                if verbose:
                    print(f"  [NL] Problema solver lineare allo step {step}, analisi interrotta.")
                    print(log_text.strip())
                break

            if ok < 0:
                if verbose:
                    print(f"  [NL] analyze failed allo step {step}")
                    print(log_text.strip())
                break

            u = ops.nodeDisp(control_node, 1)  # [m]
            ops.reactions()

            Vb = 0.0
            for (i, j), nd in node_tags.items():
                if j == 0:
                    Vb += ops.nodeReaction(nd, 1)

            disp_mm.append(u * 1000.0)
            shear_kN.append(-Vb / 1000.0)

            if disp_mm and disp_mm[-1] >= target_mm * 1.0:
                break

        disp_arr = np.array(disp_mm, dtype=float)
        shear_arr = np.array(shear_kN, dtype=float)

        if j2_problem or lin_problem:
            return {
                "status": "error",
                "message": "analysis_not_converged",
                "disp_mm": disp_arr.tolist(),
                "shear_kN": shear_arr.tolist(),
                "V_target": None
            }

        V_target = shear_at_target_disp(disp_arr, shear_arr, target_mm)

        return {
            "status": "ok",
            "message": None,
            "disp_mm": disp_arr.tolist(),
            "shear_kN": shear_arr.tolist(),
            "V_target": V_target
        }

    except Exception as e:
        if verbose:
            print(f"[run_pushover_nonlinear] errore: {e}")
        return {
            "status": "error",
            "message": str(e),
            "disp_mm": [],
            "shear_kN": [],
            "V_target": None
        }


# ============================================================
#  FUNZIONE ALTO LIVELLO: EXISTING + PROJECT
# ============================================================

def run_two_cases_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    data deve contenere:
      {
        "existing_openings": [[x1,x2,y1,y2], ...],
        "project_openings":  [[x1,x2,y1,y2], ...]
      }
    """
    existing_openings = [tuple(o) for o in data.get("existing_openings", [])]
    project_openings  = [tuple(o) for o in data.get("project_openings", [])]

    if not existing_openings:
        raise ValueError("Chiave 'existing_openings' mancante o vuota.")
    if not project_openings:
        raise ValueError("Chiave 'project_openings' mancante o vuota.")

    res_existing = run_pushover_nonlinear(existing_openings)
    res_project  = run_pushover_nonlinear(project_openings)

    return {
        "existing": res_existing,
        "project": res_project
    }


# ============================================================
#  FASTAPI APP
# ============================================================

app = FastAPI(
    title="Wall Pushover Service (Light)",
    description="Servizio pushover muratura (Existing + Project) – solo curva forza-spostamento",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Wall Pushover Service (light) attivo"}


@app.post("/pushover")
async def pushover_endpoint(request: Request):
    """
    Endpoint principale.
    Accetta JSON nel formato:
      - [{ "existing_openings": [...], "project_openings": [...] }]
        (come n8n/Lovable nel tuo caso)
      - oppure { "existing_openings": [...], "project_openings": [...] }
    """
    payload = await request.json()

    # gestisce sia lista che dict
    if isinstance(payload, list):
        if not payload:
            return JSONResponse(
                status_code=400,
                content={"error": "Payload list vuota."}
            )
        payload = payload[0]

    try:
        result = run_two_cases_from_dict(payload)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
