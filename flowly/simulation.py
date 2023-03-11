import copy
from typing import cast
from chemicals import R, mixing_simple
from thermo import EquilibriumState
from fluids.compressible import T_critical_flow, P_critical_flow
import SU2 as su2
from ezmesh import CurveLoop, PlaneSurface, Geometry, visualize_mesh
import numpy as np

from diffuse.diffusers import Diffuser


def get_mesh(diffuser: Diffuser):
    with Geometry() as geo:
        diffuser_coords = np.concatenate([
            diffuser.shroud_line,
            diffuser.bottom_line[::-1],
        ])
        diffuser_curve_loop = CurveLoop(
            diffuser_coords,
            mesh_size=0.05,
            labels={
                "wall":  [0, 2],
                "inflow":  [3],
                "outflow": [1],
            },
            transfinite_cell_counts={
                50: [0, 1, 2, 3]
            }
        )
        surface = PlaneSurface(diffuser_curve_loop, is_quad_mesh=True, transfinite_corners=[0, 1, 2, 3])
        mesh = geo.generate(surface)
        geo.write("./meshes/NICFD_nozzle.su2")

        return mesh
    
        # visualize_mesh(mesh)


def get_config(inlet_total_state: EquilibriumState, inlet_mach_number: float):
    gamma = cast(float, inlet_total_state.Cp_Cv_ratio())  # type: ignore
    molar_weight = cast(float, mixing_simple(inlet_total_state.zs, inlet_total_state.constants.MWs))                                                    # type: ignore
    specific_gas_constant = R * 1000 / molar_weight
    inlet_static_pressure = inlet_total_state.P/(1+((gamma-1)/2)*inlet_mach_number**2)**(gamma/(gamma-1))

    return su2.io.Config({
        "SOLVER": "RANS",
        "KIND_TURB_MODEL": "SST",
        "MATH_PROBLEM": "DIRECT",
        "RESTART_SOL": "NO",
        "SYSTEM_MEASUREMENTS": "SI",
        "MACH_NUMBER": inlet_mach_number,
        "AOA": 0.0,
        "SIDESLIP_ANGLE": 0.0,
        "INIT_OPTION": "TD_CONDITIONS",
        "FREESTREAM_OPTION": "TEMPERATURE_FS",
        "FREESTREAM_PRESSURE": inlet_total_state.P,
        "FREESTREAM_TEMPERATURE": inlet_total_state.T,
        "REF_DIMENSIONALIZATION": "DIMENSIONAL",
        "FLUID_MODEL": "PR_GAS",
        "GAMMA_VALUE": gamma,
        "GAS_CONSTANT": specific_gas_constant,
        "CRITICAL_TEMPERATURE": T_critical_flow(inlet_total_state.T, gamma),
        "CRITICAL_PRESSURE": P_critical_flow(inlet_total_state.P, gamma),
        "ACENTRIC_FACTOR": inlet_total_state.pseudo_omega(),
        "VISCOSITY_MODEL": "CONSTANT_VISCOSITY",
        "MU_CONSTANT": inlet_total_state.mu(),                                  # type: ignore
        "CONDUCTIVITY_MODEL": "CONSTANT_CONDUCTIVITY",
        "THERMAL_CONDUCTIVITY_CONSTANT": inlet_total_state.k(),                 # type: ignore
        "MARKER_HEATFLUX": "( wall, 0.0 )",
        "MARKER_SYM": "SYMMETRY",
        "MARKER_RIEMANN": f"( inflow, TOTAL_CONDITIONS_PT, {inlet_total_state.P}, {inlet_total_state.T}, 1.0, 0.0, 0.0, outflow, STATIC_PRESSURE, {inlet_static_pressure}, 0.0, 0.0, 0.0, 0.0 )",
        "NUM_METHOD_GRAD": "GREEN_GAUSS",
        "MARKER_MONITORING": "wall",
        "CFL_NUMBER": 10.0,
        "CFL_ADAPT": "YES",
        "CFL_ADAPT_PARAM": "( 0.1, 2.0, 10.0, 1000.0 )",
        "MAX_DELTA_TIME": 1E6,
        "MUSCL_FLOW": "YES",
        "SLOPE_LIMITER_FLOW": "NONE",
        "MUSCL_TURB": "NO",
        "LINEAR_SOLVER": "FGMRES",
        "LINEAR_SOLVER_PREC": "ILU",
        "LINEAR_SOLVER_ILU_FILL_IN": 0,
        "LINEAR_SOLVER_ERROR": 1E-6,
        "LINEAR_SOLVER_ITER": 10,
        "MGLEVEL": 0,
        "CONV_NUM_METHOD_FLOW": "ROE",
        "ENTROPY_FIX_COEFF": 0.1,
        "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
        "CONV_NUM_METHOD_TURB": "SCALAR_UPWIND",
        "TIME_DISCRE_TURB": "EULER_IMPLICIT",
        "CFL_REDUCTION_TURB": 1.0,
        "ITER": 1000,
        "CONV_RESIDUAL_MINVAL": -24,
        "CONV_STARTITER": 10,
        "MESH_FILENAME": "../meshes/NICFD_nozzle.su2",
        "MESH_FORMAT": "SU2",
        "MESH_OUT_FILENAME": "mesh_out.su2",
        "SOLUTION_FILENAME": "solution_flow.dat",
        "SOLUTION_ADJ_FILENAME": "solution_adj.dat",
        "TABULAR_FORMAT": "CSV",
        "CONV_FILENAME": "history",
        "RESTART_FILENAME": "restart_flow.dat",
        "VOLUME_FILENAME": "flow",
        "SURFACE_FILENAME": "surface_flow",
        "OUTPUT_WRT_FREQ": 1000,
        "SCREEN_OUTPUT": "(INNER_ITER, RMS_DENSITY, RMS_TKE, RMS_DISSIPATION, LIFT, DRAG)",

        # Defaults values for SU2 Python to run without error
        "HISTORY_OUTPUT": ['ITER', 'RMS_RES'],
        "DV_KIND": ['NO_DEFORMATION'],
        "DEFINITION_DV": {'FFDTAG': [[]], 'KIND': ['HICKS_HENNE'], 'MARKER': [['WING']], 'PARAM': [[0.0, 0.05]], 'SCALE': [1.0], 'SIZE': [1]},
        "DV_PARAM": {'FFDTAG': [[]], 'PARAM': [[0]]},
        "DV_VALUE_NEW": [0.0],
        "DV_VALUE_OLD": [0.0],
        "NUMBER_PART": 0,
        "OPT_OBJECTIVE": {'DRAG': {'OBJTYPE': 'DEFAULT', 'SCALE': 1.0}},
        "NZONES": 1,
    })


def run_simulation(inlet_total_state: EquilibriumState, inlet_mach_number: float):
    # load config and start state
    config = get_config(inlet_total_state, inlet_mach_number)
    state = su2.io.State()

    # find solution files if they exist
    state.find_files(config)

    # start results data
    results = su2.util.bunch()

    # konfig = copy.deepcopy(config)
    # konfig.DISCARD_INFILES = 'YES'
    # zstate = copy.deepcopy(state)
    drag = su2.eval.func('DRAG', config, state)
