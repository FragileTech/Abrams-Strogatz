from typing import Union

import panel as pn
import param

from language_competition.models import AbramsStrogatz, MinettWang


class WidgetPanel(param.Parameterized):
    """Create an interactive panel representing the evolution of the grid."""

    model_selector = param.ObjectSelector(
        default="Abrams-Strogatz", objects=["Abrams-Strogatz", "Minett-Wang"]
    )
    width = param.Integer(default=20, bounds=(5, 100))
    height = param.Integer(default=20, bounds=(5, 100))
    shape = param.Range(default=(20, 20), bounds=(5, 100))
    n_iterations = param.Integer(default=1500, bounds=(100, 5000))
    volatility = param.Number(1.0, bounds=(0.5, 50))
    status_a = param.Number(0.5, bounds=(0, 1))
    prob_a = param.Number(0.5, bounds=(0, 1))
    prob_b = param.Number(0.3, bounds=(0, 1), precedence=-1)
    step = param.Selector(objects=[0])

    def __init__(self):
        """
        Instantiate the class. The instance attribute can be selected
        by using an interactive panel.
        """
        super(WidgetPanel, self).__init__()
        self.grid_instance = None
        self.compute_grid_instance()

    @param.depends("height", "width", watch=True)
    def _update_shape(self) -> None:
        """Update the shape of the lattice when changing height, width parameters."""
        self.param["shape"].default = (self.height, self.width)
        self.shape = (self.height, self.width)

    @param.depends("model_selector", watch=True)
    def _update_precedence(self) -> None:
        """Update prob_b precedence to 1 when model_selector is 'Minett-Wang'."""
        model = self.model_selector
        if model == "Minett-Wang":
            self.param["prob_b"].precedence = 1
            self.prob_b = 0.3
            self.prob_a = 0.3
        else:
            self.param["prob_b"].precedence = -1
            self.prob_a = 0.5

    @param.depends(
        "shape", "volatility", "status_a", "prob_a", "prob_b", "model_selector", watch=True
    )
    def compute_grid_instance(self) -> None:
        """
        Compute the evolution of the grid following the Abrams-Strogatz or
        Minett-Wang model.

        Any time the instance parameters are changed, the model should be
        recalculated.
        """
        selected_model = self.model_selector
        if selected_model == "Abrams-Strogatz":
            model_name = AbramsStrogatz
        elif selected_model == "Minett-Wang":
            model_name = MinettWang
        else:
            raise TypeError("Introduced model_selector value is not valid")
        # Compute lattice using the selected model
        self.grid_instance = model_name(
            shape=self.shape,
            status_a=self.status_a,
            vol=self.volatility,
            prob_a0=self.prob_a,
            prob_b0=self.prob_b,
        )
        # Run the model iterate_instance_model
        self.iterate_instance_model()

    @param.depends("n_iterations", watch=True)
    def iterate_instance_model(self):
        """Iterate the model 'n_iterations' steps."""
        _ = self.grid_instance.run(epochs=self.n_iterations)
        self.param["step"].objects = list(range(0, len(self.grid_instance.memory)))

    @param.depends("step")
    def view(self):
        """Represent the grid according to the grid_plot method of each class."""
        return self.grid_instance.grid_plot(self.step)

    def widget_panel(self) -> pn.Column:
        """
        Generate a widget to visualize the evolution of the lattice.

        This method creates a widget that shows the evolution of the lattice
        during the iterative process. To view this evolution, the method
        'run' should have been called (in order to have a complete
        cycle). Otherwise, the attribute self.memory is empty.
        """
        # Widget
        panel_widget = {
            "step": {
                "widget_type": pn.widgets.DiscretePlayer,
                "interval": 1,
                "loop_policy": "once",
                "value": 0,
            }
        }
        return pn.Column(
            self.view, pn.Param(self.param["step"], widgets=panel_widget, name="Iteration")
        )

    def interactive_panel(self) -> pn.Row:
        """
        Generate the final product. Instance attributes can be set using the interactive panel
        """
        return pn.Row(
            pn.Column(
                pn.panel(
                    self.param,
                    parameters=[
                        "model_selector",
                        "width",
                        "height",
                        "n_iterations",
                        "volatility",
                        "status_a",
                        "prob_a",
                        "prob_b",
                    ],
                )
            ),
            self.widget_panel(),
        )
