# %%
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import numpy as np
from paraflow.flow_station import FlowStation
import numpy as np
from paraflow import SymmetricPassage, run_simulation, FlowStation
from pymoo.core.problem import ElementwiseProblem

from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize



MaxOrMin = Literal["max", "min"]


@dataclass
class OptimizationSpecification:
    inflow: FlowStation
    inlet_radius: float
    num_ctrl_pts: int
    objectives: List[Tuple[Literal["mach"], MaxOrMin]]
    area_ratio: Optional[float] = None


class PassageOptimizationProblem(ElementwiseProblem):
    def __init__(self, spec: OptimizationSpecification):
        self.spec = spec

        prop_lower_bound = np.zeros(self.spec.num_ctrl_pts)
        prop_upper_bound = np.ones(self.spec.num_ctrl_pts)

        angles_lower_bound = np.full(self.spec.num_ctrl_pts, -np.pi/2)
        angles_upper_bound = np.full(self.spec.num_ctrl_pts, 2*np.pi)

    # contour_props=[0.25, 0.25, 0.35, 0.5, 0.5, 0.75],
    # contour_angles=np.radians([-15.0, -15.0, 0.0, 3, 5.0, 15.0]).tolist()

        super().__init__(
            n_var=self.spec.num_ctrl_pts,
            n_obj=len(self.spec.objectives),
            n_ieq_constr=0,
            # xl=0,
            # xu=1
            xl=np.concatenate([prop_lower_bound, angles_lower_bound]),
            xu=np.concatenate([prop_upper_bound, angles_upper_bound]),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        # passage = SymmetricPassage(
        #     inlet_radius=self.spec.inlet_radius,
        #     area_ratio=self.spec.area_ratio,
        #     axial_length=1,
        #     contour_props=x[:, :self.spec.num_ctrl_pts].tolist(),
        #     contour_angles=x[:, self.spec.num_ctrl_pts:].tolist(),
        # )

        # sim_results = run_simulation(passage, self.spec.inflow, "/workspaces/paraflow/simulation")
        # objectives = []
        # for obj, direction in self.spec.objectives:
        #     sign = -1 if direction == "max" else 1
        #     if obj == "mach":
        #         obj_val = sim_results["mid_outflow"].mach_number
        #     else:
        #         raise ValueError(f"Unknown objective {obj}")
        #     objectives.append(sign * obj_val)

        out["F"] = np.column_stack([0])
        out["G"] = np.column_stack([])


def optimize(spec: OptimizationSpecification):
    problem = PassageOptimizationProblem(spec)
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=1.0, eta=3, vtype=float),
        mutation=PM(prob=1.0, eta=3, vtype=float),
        eliminate_duplicates=True
    )


    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 40),
        seed=1,
        save_history=True,
        verbose=True
    )
    # with open('optimization.pkl', 'wb') as optimization_result_file:
    #     pickle.dump(res, optimization_result_file, pickle.HIGHEST_PROTOCOL)
