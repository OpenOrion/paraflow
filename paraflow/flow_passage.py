import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

class FlowPassage:
    def __init__(
        self,
        axial_length: float,
        top_line: npt.NDArray[np.float64],
        bottom_line: npt.NDArray[np.float64],
    ) -> None:
        self.symetry_line = np.vstack(
            [
                [0, 0],
                [axial_length, 0]
            ]
        )

        self.top_line = top_line
        self.bottom_line = bottom_line

    def visualize(self):
        fig = go.Figure(
            layout=go.Layout(title=go.layout.Title(text="Annular Diffuser"))
        )
        coords = self.get_coords()
        fig.add_trace(go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            name=f"Passage"

        ))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        fig.show()

    def get_coords(self) -> npt.NDArray[np.float64]: return np.array([])

