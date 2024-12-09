# Turbulent/Laminar Mixing Simulator
A python code for fast real-time 2D simulations of scalar mixing in turbulent or laminar flows for various boundary conditions.
![image](https://github.com/jorishey1234/mixing_simulator/assets/35989752/f4cb94a8-dabd-446b-b001-7e2608a82973)

The code solves the advection-diffusion equation for a synthetic flow reproducing turbulent spectrum properties. The flow consist in an alternation of horizontal and vertical waves, in the manner of the sine flow, with tunable spectral properties (see parameters below).

You can interact with the simulation by dropping patches of scalar with mouse clicking on the image, right or left.
![image](https://github.com/jorishey1234/mixing_simulator/assets/35989752/73f06571-6039-42c2-b929-5832484b3720)

# Install
Install necessary python packages with pip/pip3 :

>> pip3 install numpy time PyQt5 vispy scipy argparse

# Run
Quick run :
>> python3 mixing_simulator_v1.0.py

You can interact with the simulation by dropping patches of scalar with mouse clicking on the image, right or left.

Set grid size $n \times n$
>> python3 mixing_simulator_v1.0.py -n 512

Set horizontal grid periodicity $n \times p \cdot n$
>> python3 mixing_simulator_v1.0.py -n 512 -p 3

# Simulation parameters :
Simulation Mode :
- Gradient : Simulate scalar mixing in a mean horizontal scalar gradient $c' = c- \overline{\nabla c} \cdot x$
- Decay : Simulate decay of scalar fluctuations (periodic boundaries)
- Source : Simulate mixing of a scalar plume with point source and mean horizontal flow $\langle u \rangle = - 2 A/t_c$

Flow properties :

- Roughness : Flow velocity roughness, measure by the slope of the velocity lengthscale spectrum: $\langle v(k)^2 \rangle\sim k^{\xi}$. $\xi=-5/3$ in 3D turbulence, $\xi=-3$  in 2D turbulence, $\xi=-\infty$ in single lengthscale laminar flows.
- Max lengthscale : Maximum velocity lengthscale (domain size is 1)
- Min lengthscale : Minimum velocity lengthscale
- Correlation time : Velocity correlation time $t_c$ of the largest length scale $L=1$. The correlation time of smaller scales $\ell$ is $t_c'(\ell) \sim t_c \ell$
- Amplitude : Amplitude $A$ of the velocity fluctuations. $\langle v^2 \rangle ^{1/2} = A / t_c$

Scalar properties :

- Diffusion : Molecular diffusivity of the scalar

Simultaion properties :

- FPS : Frame per second on screen. Max possible fps (depending on hardware config) is indicated in the console.
