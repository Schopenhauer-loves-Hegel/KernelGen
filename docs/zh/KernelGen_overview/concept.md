# 概念

本节列出了 Kernel 生成的基本概念：

- **正确性**：生成的 Kernel 的输出在数值上与 PyTorch 基准测试的匹配程度。KernelGen 会在每个场景中比较 Kernel 和 PyTorch 基准测试，并输出总体正确性。只有正确性通过，该 Kernel 才能被使用。

- **加速比**：生成的 Kernel 与 PyTorch 基准测试相比运行快多少。Kernel 相对于 PyTorch 基准测试的加速比是 PyTorch 执行时间与 Kernel 执行时间的比值。

- **场景**：输入参数的特定组合。每个独特的组合对应一个不同的生成 Kernel。例如，如果输入参数包括两个张量形状和两种数据类型，那么就有四个场景。
