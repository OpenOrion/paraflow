# from dataclasses import dataclass
# import numpy as np
# from pymoo.optimize import minimize
# from pymoo.core.problem import ElementwiseProblem
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.factory import get_sampling, get_crossover, get_mutation
# from pymoo.factory import get_termination

# @dataclass
# class PassageObjectives:
#     pass

# @dataclass
# class PassageConstraints:
#     pass

# @dataclass
# class OptimizationSpecification:
#     fluid: PassageFluid
#     objectives: PassageObjectives
#     constraints: PassageConstraints

# class PassageOptimizationProblem(ElementwiseProblem):
#     def __init__(self, spec: OptimizationSpecification):


#         super().__init__(
#             n_var=len(bounds),
#             n_obj=1,
#             n_constr=29,
#             xl=[bound.min for bound in bounds],
#             xu=[bound.max for bound in bounds]
#         )

#     def _evaluate(self, x, out, *args, **kwargs):
#         out["F"] = np.column_stack([])
#         out["G"] = np.column_stack([])


# def optimize(spec: OptimizationSpecification):
#     problem = PassageOptimizationProblem(spec)
#     algorithm = NSGA2(
#         pop_size=100,
#         n_offsprings=10,
#         sampling=get_sampling("real_random"),
#         crossover=get_crossover("real_sbx", prob=0.9, eta=15),
#         mutation=get_mutation("real_pm", eta=20),
#         eliminate_duplicates=True
#     )

#     termination = get_termination("n_gen", 100)

#     res = minimize(
#         problem,
#         algorithm,
#         termination,
#         seed=1,
#         save_history=True,
#         verbose=True
#     )
#     # with open('optimization.pkl', 'wb') as optimization_result_file:
#     #     pickle.dump(res, optimization_result_file, pickle.HIGHEST_PROTOCOL)
