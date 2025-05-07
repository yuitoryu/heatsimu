# 2D Heat Equation simulator #
This project focuses on plotting heat transfer for the following heat equation:  
$\begin{cases}
        u_t = \frac{\partial}{\partial x}[c(x,y)u_x] + \frac{\partial}{\partial y}[c(x,y)u_y] + Q(x,y,t) \\
        a_{left}u(0,y,t) + b_{left}(0,y,t) = \phi_{left}(y,t) \\
        a_{right}u(L_x,y,t) + b_{right}(L_x,y,t) = \phi_{right}(y,t) \\
        a_{up}u(x,0,t) + b_{up}(x,0,t) = \phi_{up}(x,t) \\
        a_{down}u(x,L_y,t) + b_{down}(x,L_y,t) = \phi_{down}(x,t) \\
        u(x,y,0) = \psi(x,y)
    \end{cases}$  
There are 3 important classes that you need to import from h2d.py for simulation.  
### BC_2D ###
This class takes care of the boundary condition. `BC_2D(left, right, up, down)`. All inputs are mandatory and follow a format of `(alpha, beta, f(x,y,t))`. Please note that although in the above equation the input functions should only depends on 2 variables, please input functions with 3 variables.  

### ContConduct ###
This class handles non-constant conductivity. `ContConduct(c_func)`. The input function should be dependent on $x$ and $y$ only. For performance concerns, if your are using constant conductivity, it is recommended to use that num,ber directly instead of this class.

### Heat2dSimu ###
This is the simulator class itself. The following are mandatory inputs.  
`map_shape (tuple)`: Physical size of the 2D domain.  
`dx (number types supported by Pytorch)`: Step size of *interior* points (excluding boundaries).  
`total_time (number types supported by Pytorch)`: End time for the simulation.  
`dt (number types supported by Pytorch)`: Step size of time.  
`bc (BC_2D)`: Boundary condition with 4 elements. Order: up, r down, left, right.  
`ic (callable(x,y))`: Function for initial condition.  
`c (ContConduct or number types supported by Pytorch)`: Diffusion coefficient.  
`plot_step (int)`: How often (in steps) to plot the solution.  
The following are optional inputs.  
`Q (callable(x,y,t) or number types supported by Pytorch)`: heat source.  
`device (str)`: 'cpu' or 'cuda', which device to use for Tensor operations.  
`do_progress_bar (bool)`: whether to disable progress bar during simulation.  
`dtype (torch.dtype)`: data type for simulation. Strongly recommend not to use float16. It give shit results.  
You can initialize a simulator by:  
```bash
test = Heat2dSimu(map_shape, dx, total_time, dt, bc, ic, c, plot_step, Q, device='cuda', do_progress_bar=True, dtype=torch.float32)
```
To start a simulation session, do:
```bash
test.start()
```

## Notes ##
If you want to see the animation, run the code in Visual Studio Code. It does not work in PyCharm since it only plot the first frame.  
`triton` is necessary since `@torch.compile` is frequently used. To install `triton`, go to `https://huggingface.co/madbuda/triton-windows-builds`.

