# %%
from dataclasses import dataclass
import multiprocessing
import pickle
from typing import List, Literal, Optional, Tuple
import numpy as np
from paraflow.flow_station import FlowStation
import numpy as np
from paraflow import FlowStation
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization

from paraflow.passages.symmetric import SymmetricPassage
from paraflow.simulation import run_simulation
import ray
import os

MaxOrMin = Literal["max", "min"]
n_proccess = 1
pool = multiprocessing.Pool(n_proccess)
runner = StarmapParallelization(pool.starmap)


@dataclass
class OptimizationSpecification:
    inflow: FlowStation
    inlet_radius: float
    num_ctrl_pts: int
    num_throat_pts: int
    objectives: List[Tuple[Literal["mach"], MaxOrMin]]
    area_ratio: Optional[float] = None


class PassageOptimizationProblem(ElementwiseProblem):

    def __init__(self, spec: OptimizationSpecification):
        self.spec = spec
        self.iteration = 0
        self.variable_config = {
            "contour_props": np.tile([0,1], (self.spec.num_ctrl_pts,1)),
            "contour_angles": np.tile([0, np.pi/2], (self.spec.num_ctrl_pts,1)),
        }
        bounds = np.concatenate(list(self.variable_config.values()), axis=0)
        super().__init__(
            n_var=self.spec.num_ctrl_pts*2,
            n_obj=len(self.spec.objectives),
            n_ieq_constr=1,
            xl=bounds[:, 0],
            xu=bounds[:, 1],
            elementwise_runner=runner
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
            contour_angles[:self.spec.num_throat_pts] = -contour_angles[:self.spec.num_throat_pts] 

            passage = SymmetricPassage(
                inlet_radius=self.spec.inlet_radius,
                area_ratio=self.spec.area_ratio,
                axial_length=1,
                contour_props=contour_props.tolist(),
                contour_angles=contour_angles.tolist(),
            )

            assert (passage.ctrl_pnts[:, 1] > 0).all()

            remote_result = run_simulation.remote(passage, self.spec.inflow, "/workspace/simulation")
            sim_results = ray.get(remote_result)
            objectives = []
            for obj, direction in self.spec.objectives:
                sign = -1 if direction == "max" else 1
                if obj == "mach":
                    obj_val = sim_results["mid_outflow"].mach_number
                else:
                    raise ValueError(f"Unknown objective {obj}")
                objectives.append(sign * obj_val)
            print(objectives)
            passage.visualize(f"nozzle{self.iteration}", show=False, save_path=f"./output/nozzle{self.iteration}")
            os.rename('./simulation/flow.vtu', f'./simulation/flow{self.iteration}.vtu')

            self.iteration += 1

        except:
            objectives = np.zeros(len(self.spec.objectives))
            is_valid = False

        out["F"] = objectives
        out["G"] = [int(not is_valid)]

def optimize(spec: OptimizationSpecification):
    problem = PassageOptimizationProblem(spec)

    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ("n_gen", 10000),
                   seed=1,
                   save_history=True,
                   verbose=True)

    # X, F = res.opt.get("X", "F")

    with open('optimization.pkl', 'wb') as optimization_result_file:
        pickle.dump(res, optimization_result_file, pickle.HIGHEST_PROTOCOL)
