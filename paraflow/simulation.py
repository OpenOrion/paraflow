import pysu2
from mpi4py import MPI
from ezmesh.exporters import export_to_su2
from paraflow.passages.common import Passage, PassageFluid

CONFIG_FILE_NAME = "config.cfg"


def setup_simulation(
    working_directory: str,
    passage: Passage,
    fluid: PassageFluid
):
    config = passage.get_config(fluid)
    mesh = passage.get_mesh()
    with open(f"{working_directory}/{CONFIG_FILE_NAME}", "w") as f:
        for key, value in config.items():
            if key == "MESH_FILENAME":
                value = f"{working_directory}/{value}"
            f.write(f"{key}= {value}\n")
        export_to_su2(mesh, f"{working_directory}/{config['MESH_FILENAME']}")


def run_simulation(
    working_directory: str,
    passage: Passage,
    fluid: PassageFluid
):
    setup_simulation(working_directory, passage, fluid)

    # Import mpi4py for parallel run
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize the corresponding driver of SU2, this includes solver preprocessing
    SU2Driver = pysu2.CSinglezoneDriver(f"{working_directory}/{CONFIG_FILE_NAME}", 1, comm)

    MarkerID = None
    MarkerName = 'wall'       # Specified by the user

    # Get all the boundary tags
    MarkerList = SU2Driver.GetMarkerTags()

    # Get all the markers defined on this rank and their associated indices.
    allMarkerIDs = SU2Driver.GetMarkerIndices()

    # Check if the specified marker exists and if it belongs to this rank.
    if MarkerName in MarkerList and MarkerName in allMarkerIDs.keys():
        MarkerID = allMarkerIDs[MarkerName]

    # Number of vertices on the specified marker (per rank)
    nVertex_Marker = 0  # total number of vertices (physical + halo)

    if MarkerID != None:
        nVertex_Marker = SU2Driver.GetNumberMarkerNodes(MarkerID)
    print(nVertex_Marker)
    # primitiveIndices = SU2Driver.GetPrimitiveIndices() # maps primitive names to their indices.
    # temperatureIndex = primitiveIndices["TEMPERATURE"]
    # primitives = SU2Driver.Primitives()

    # for iVertex in range(nVertex_Marker):
    #   # SU2Driver.SetMarkerCustomDisplacement(MovingMarkerID, int(iVertex), value)
    #   # print(primitives)
    #   # fxyz = SU2Driver.GetMarkerFlowLoad(MarkerID, iVertex)
    #   # print(fxyz)
    #   x, y = SU2Driver.InitialCoordinates().Get(iVertex)
    #   print(x,y)
    #   print(primitives(iVertex, temperatureIndex))

    # Time loop is defined in Python so that we have acces to SU2 functionalities at each time step
    comm.Barrier()

    # Time iteration preprocessing
    SU2Driver.Preprocess(0)

    # Run one time-step (static: one simulation)
    SU2Driver.Run()

    # Postprocess
    SU2Driver.Postprocess()

    # Update the solver for the next time iteration
    SU2Driver.Update()

    # Monitor the solver and output solution to file if required
    SU2Driver.Monitor(0)

    # Output the solution to file
    SU2Driver.Output(0)

    # Finalize the solver and exit cleanly
    SU2Driver.Finalize()
