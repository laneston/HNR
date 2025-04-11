import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph
from efficientnet_b0 import MNISTEfficientNet


class ModelVisualizer:
    def __init__(self, model, input_size=(1, 1, 224, 224)):
        self.model = model
        self.input_size = input_size
        self.writer = SummaryWriter()  # TensorBoard日志写入器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def visualize(self):
        """执行可视化流程"""
        # 生成简化计算图
        self._generate_simplified_graph()

        # 生成TensorBoard详细日志
        self._generate_tensorboard_log()

        # 关闭写入器
        self.writer.close()

    def _generate_simplified_graph(self):
        """生成并保存简化计算图"""
        # 使用torchview生成可视化图形
        model_graph = draw_graph(
            self.model,
            input_size=self.input_size,
            # device="meta",
            device=self.device.type,
            expand_nested=True,  # 展开嵌套结构
            hide_module_functions=False,  # 显示模块函数
            depth=10,  # 可视化深度
        )

        # 保存为矢量图
        model_graph.visual_graph.render(
            filename="./model/model_architecture", format="svg", cleanup=True
        )
        print("简化计算图已保存为 model_architecture.svg")

    def _generate_tensorboard_log(self):
        """生成TensorBoard日志"""
        # 创建虚拟输入
        dummy_input = torch.randn(self.input_size).to(self.device)

        # 添加计算图到TensorBoard
        self.writer.add_graph(self.model, dummy_input)
        print("TensorBoard日志已生成，使用以下命令查看：")
        print("tensorboard --logdir=runs")


if __name__ == "__main__":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # 初始化模型
    model = MNISTEfficientNet().model

    # 创建可视化实例
    visualizer = ModelVisualizer(
        model=model, input_size=(1, 1, 224, 224)  # 批大小×通道×高×宽
    )

    # 执行可视化
    visualizer.visualize()
