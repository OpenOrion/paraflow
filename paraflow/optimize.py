from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple, Union
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from paraflow.flow_station import FlowStation
import numpy as np
from paraflow import SymmetricPassage, run_simulation, FlowStation



MaxOrMin = Literal["max", "min"]
GreaterOrLess = Literal["greater", "less", "equal"]
PositiveOrNegative = Literal["positive", "negative"]


    # contour_props=[0.25, 0.25, 0.35, 0.5, 0.5, 0.75],
    # contour_angles=np.radians([-15.0, -15.0, 0.0, 3, 5.0, 15.0]).tolist()

@dataclass
class OptimizationSpecification:
    inflow: FlowStation
    inlet_radius: float
    num_ctrl_pts: int
    objectives: List[Tuple[str, MaxOrMin]]
    area_ratio: Optional[float] = None

class PassageOptimizationProblem(ElementwiseProblem):
    def __init__(self, spec: OptimizationSpecification):
        self.spec = spec



        super().__init__(
            n_var=self.spec.num_ctrl_pts,
            n_obj=len(self.spec.objectives),
            n_constr=29,
            xl=[
                
            ],
            xu=[
            
            ]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        passage = SymmetricPassage(
            inlet_radius=self.spec.inlet_radius,
            area_ratio=self.spec.area_ratio,
            axial_length=1,
            contour_props=[0.2, 0.5, 0.75],
            contour_angles=np.radians([-20, 15.0, 15.0]).tolist()
        )

        sim_results = run_simulation(passage, self.spec.inflow, "/workspaces/paraflow/simulation")
        objectives = []
        for obj, direction in self.spec.objectives:
            sign = -1 if direction == "max" else 1
            if obj == "mach":
                obj_val = sim_results["mid_outflow"].mach_number
            else:
                raise ValueError(f"Unknown objective {obj}")
            objectives.append(sign * obj_val)

        out["F"] = np.column_stack(objectives)
        out["G"] = np.column_stack([])


def optimize(spec: OptimizationSpecification):
    problem = PassageOptimizationProblem(spec)
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 100)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True
    )
    # with open('optimization.pkl', 'wb') as optimization_result_file:
    #     pickle.dump(res, optimization_result_file, pickle.HIGHEST_PROTOCOL)
