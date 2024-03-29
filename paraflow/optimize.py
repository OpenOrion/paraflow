import multiprocessing
import pickle
from typing import List, Literal, Optional, Tuple, cast
import numpy as np
import numpy as np
from paraflow import FlowState
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from paraflow.simulation.postprocessing import get_point_data
from paraflow.simulation.simulation import run_simulation
from paraflow.passages import SymmetricPassage, SimulationParams
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

MaxOrMin = Literal["max", "min"]
# n_proccess = 1
# pool = multiprocessing.Pool(n_proccess)
# runner = StarmapParallelization(pool.starmap)


class PassageOptimizer(ElementwiseProblem):

    def __init__(
        self,
        working_directory: str,
        inflow: FlowState,
        outlet_static_state: FlowState,
        inlet_radius: float,
        num_ctrl_pts: int,
        num_throat_pts: int,
        objectives: List[Tuple[Literal["mach"], MaxOrMin]],
        area_ratio: Optional[float] = None,
    ):
        self.working_directory = working_directory
        self.inlet_total_state = inflow
        self.outlet_static_state = outlet_static_state
        self.inlet_radius = inlet_radius
        self.num_ctrl_pts = num_ctrl_pts
        self.num_throat_pts = num_throat_pts
        self.objectives = objectives
        self.area_ratio = area_ratio

        self.iteration = 0
        self.variable_config = {
            "contour_props": np.tile([0, 1], (self.num_ctrl_pts, 1)),
            "contour_angles": np.tile([0, np.pi/2], (self.num_ctrl_pts, 1)),
        }
        bounds = np.concatenate(list(self.variable_config.values()), axis=0)
        super().__init__(
            n_var=self.num_ctrl_pts*2,
            n_obj=len(self.objectives),
            n_ieq_constr=1,
            xl=bounds[:, 0],
            xu=bounds[:, 1],
            # elementwise_runner=runner
        )

    def _evaluate(self, x, out, *args, **kwargs):
        is_valid = True
        try:
            variable_values = {}
            variable_offset = 0
            for variable_name, bounds in self.variable_config.items():
                variable_values[variable_name] = x[variable_offset:variable_offset+bounds.shape[0]]
                variable_offset += bounds.shape[0]

            sort_idx = np.argsort(variable_values["contour_props"])
            contour_props = variable_values["contour_props"][sort_idx]
            contour_angles = variable_values["contour_angles"][sort_idx]
            contour_angles[:self.num_throat_pts] = -contour_angles[:self.num_throat_pts]

            passage = SymmetricPassage(
                inlet_radius=self.inlet_radius,
                area_ratio=self.area_ratio,
                axial_length=1,
                contour_props=contour_props.tolist(),
                contour_angles=contour_angles.tolist(),
            )

            mesh = passage.get_meshes()[0]
            mid_outflow_point = mesh.get_marker_point(passage.mesh_params.outflow_label, 0.5)
            passage.write(f"{self.working_directory}/passage{self.iteration}.json")

            assert (passage.ctrl_pnts[:, 1] > 0).all()

            sim_results = run_simulation(
                passage,
                config_params=SimulationParams(
                    inlet_total_state=self.inlet_total_state,
                    target_outlet_static_state=self.outlet_static_state,
                ),
                working_directory=self.working_directory,
                id=f"{self.iteration}"
            )
            passage.visualize(f"passage{self.iteration}", show=False, save_path=f"{self.working_directory}/passage{self.iteration}.png")
            objectives = []
            for obj, direction in self.objectives:
                sign = -1 if direction == "max" else 1
                if obj == "mach":
                    target_value_point_data = get_point_data(sim_results.grid, interp_points=np.array([
                        mid_outflow_point
                    ]))
                    target_values = vtk_to_numpy(target_value_point_data.GetArray("Mach"))
                    obj_val = cast(float, target_values[0])
                else:
                    raise ValueError(f"Unknown objective {obj}")
                objectives.append(sign * obj_val)

            objectives.append(0)
        except Exception as e:
            print(e)
            objectives = np.zeros(len(self.objectives))
            is_valid = False

        self.iteration += 1

        out["F"] = objectives
        out["G"] = [int(not is_valid)]

    def optimize(self):
        algorithm = NSGA2(
            pop_size=40,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        res = minimize(
            self,
            algorithm,
            ("n_gen", 10000),
            seed=1,
            save_history=True,
            verbose=True
        )

        # X, F = res.opt.get("X", "F")

        with open(f'{self.working_directory}/optimization.pkl', 'wb') as optimization_result_file:
            pickle.dump(res, optimization_result_file, pickle.HIGHEST_PROTOCOL)
