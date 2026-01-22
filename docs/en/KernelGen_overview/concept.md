# Concepts

This section lists the basic concepts for Kernel generation:

- **Correctness**: How closely the output of the generated Kernel matches the PyTorch benchmark numerically. KernelGen compares the Kernel and PyTorch benchmark in each scenario and outputs an overall correctness. Only the correctness is passed, the Kernel can be used.

- **Speedup**: How much faster the generated Kernel runs compared to a PyTorch benchmark. The speedup of a Kernel over a PyTorch benchmark is the ratio of the PyTorch execution time to the Kernel execution time.

- **Scenario**: A specific combination of input parameters. Each unique combination maps to a differently generated Kernel. For example, if input parameters include two tensor shapes and two data types, there are four scenarios.
