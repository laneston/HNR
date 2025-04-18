import torch
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph


class ModelVisualizer:
    def __init__(self, model, input_size=(1, 1, 224, 224)):
        self.model = model
        self.input_size = input_size
        self.writer = SummaryWriter()  # TensorBoard Log Writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Execute visualization process"""

    def visualize(self):

        # Generate simplified calculation diagram
        self._generate_simplified_graph()

        # Generate detailed logs for TensorBoard
        self._generate_tensorboard_log()

        # Close the writer
        self.writer.close()

    """Generate and save simplified calculation diagrams"""

    def _generate_simplified_graph(self):

        # Generate visual graphics using torchview
        model_graph = draw_graph(
            self.model,
            input_size=self.input_size,
            # device="meta",
            device=self.device.type,
            expand_nested=True,  # Expand nested structure
            hide_module_functions=False,  # Display module functions
            depth=10,  # Visual Depth
        )

        # Save as vector image
        model_graph.visual_graph.render(
            filename="model/model_architecture", format="svg", cleanup=True
        )
        print("Simplified calculation diagram has been saved as model_architecture.svg")

    """Generate TensorBoard logs"""

    def _generate_tensorboard_log(self):

        # Create virtual input
        dummy_input = torch.randn(self.input_size).to(self.device)

        # Add computational graph to TensorBoard
        self.writer.add_graph(self.model, dummy_input)
        print(
            "The TensorBoard log has been generated. Use the following command to view it:"
        )
        print("tensorboard --logdir=runs")
