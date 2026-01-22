# 工作流程

下图简要展示了 KernelGen 如何帮助您生成 Kernel。

![alt text](../assets/images/KernelGen-workflow-ch.png)

生成过程如下：

1. **收集 Kernel 信息**：用户向 KernelGen 输入语义的算子定义，例如参考 [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html#ReLU) 算子定义。KernelGen 从定义中收集算子的基本参数。
2. **搜索代码片段**：KernelGen 搜索与用户定义相似的代码片段作为参考，并提取 Kernel 参数。在此步骤中，用户可以选择是否使用搜索到的代码片段。
3. **生成 Kernel 代码和 CUDA 版基准实现代码**：KernelGen 生成 Kernel 代码及 CUDA 版基准实现代码。CUDA 版基准实现代码将作为 PyTorch 基准参考。
4. **基于 CUDA 版基准实现代码测试 Kernel**：KernelGen 基于 PyTorch 基准测试 Kernel，并输出正确性和加速比的测试结果。
