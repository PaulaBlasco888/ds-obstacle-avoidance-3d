# 3D Obstacle Avoidance with Dynamical Systems

Final lab project of *Robot's Control Architectures* (UC3M).

Paula Blasco Sánchez, Alberto Tono, Tala Issa Gammoh and Rodrigo Mariz.

In this project we learn a stable dynamical system (LPV-DS) from demonstrations and then we modulate
it around an obstacle in 3D. This way the robot repeats the demonstrated motion, goes around the
obstacle and still reaches the target. At the end we use the same controller to move a simulated
Franka Emika Panda with its Jacobian.

<p align="center">
  <img src="figures/viapoint3d_3d_trajectories.png" width="85%">
</p>

**Video:** https://youtu.be/3RWVf6ickV8 &nbsp;·&nbsp; **Report:** [report/report.pdf](report/report.pdf)

## Files

```
ds_learning/
  lpvds.py          LPV-DS: loading the data, GMM (EM + BIC), Lyapunov matrix P,
                    SDP for the A_k matrices, evaluation and simulation
  obstacle3d.py     ellipsoid obstacle: Gamma(x), normal, basis [n, e1, e2],
                    D(x) and M(x) = E D E^-1
train_lpvds.py      learns an LPV-DS with one of the Lab 1 datasets
main_3d.py          MAIN SCRIPT, gives the 3D results of the report
robot_sim.py        Franka Panda following the modulated DS (q_dot = J^+ x_dot_mod)
session2_lpvds_obstacle_2d.py   2D version with the Lab 2 obstacle package
test_modulation.py  numerical checks of the modulation
make_video.py       makes the demo video
lab2_obstacle_package/   Lab 2 package (with some small fixes, see its README)
models/             trained models (.json) and the demonstrations we used (.npz)
figures/            all the figures
report/             report (LaTeX source and PDF)
```

## Installation

You need [Miniforge](https://github.com/conda-forge/miniforge) (or any conda). From the folder of
the repo:

```bash
conda create -n DS_obstacle_avoidance -c conda-forge python=3.11 pip -y
conda activate DS_obstacle_avoidance
pip install -r requirements.txt
pip install -e lab2_obstacle_package     # only needed for the 2D script and the video
```

We installed everything with pip because some of the conda packages didn't work for us on Windows.

## How to run it

```bash
python main_3d.py              # 3D simulation, prints the results and saves the figures
python main_3d.py --animate    # the same but with a 3D animation
python robot_sim.py            # Franka Panda runs and figures
python robot_sim.py --show     # robot animation
python test_modulation.py      # checks of M(x)
python session2_lpvds_obstacle_2d.py   # 2D version (Lab 2 package + learned DS)
python make_video.py           # makes figures/demo_video.mp4 again (around 2 min)
```

The trained models are already in `models/`. If you want to train them again you need the Lab 1
datasets in `../Lab Session 1 Code-20261109/...`:

```bash
python train_lpvds.py --dataset 3D_viapoint_3 --name viapoint3d --nb_traj 6 --eps 0.01 --direction_weight 4
python train_lpvds.py --dataset 2D_Sshape --name sshape2d
```

## How it works

1. **Learned DS.** `f(x) = Σ γ_k(x) (A_k x + b_k)`. The weights γ_k come from a GMM fitted to the
   demonstrations, and the A_k from a convex SDP with `A_kᵀP + P A_k ≺ 0` and `b_k = -A_k x*`. So
   `V = (x-x*)ᵀP(x-x*)` is a Lyapunov function and every trajectory ends at x*.
2. **Obstacle.** `Γ(x) = Σ (x̃_i / (s_f a_i))²` in the frame of the ellipsoid, `n = ∇Γ/‖∇Γ‖`, and
   with it we build the orthonormal basis `[n, e1, e2]`.
3. **Modulation.** `M = E D E⁻¹` with `E = [r e1 e2]` and `D = diag(1 - 1/Γ, 1 + 1/Γ, 1 + 1/Γ)`,
   and then `ẋ = M(x) f(x)`. On the surface the part of the velocity that goes into the obstacle is
   0, and far away M is almost the identity.
4. **Robot.** Every 10 ms we compute `p = FK(q)`, `ẋ_mod = M f` and
   `q̇ = J⁺ [ẋ_mod; ω] + (I - J⁺J) q̇₀` (damped least squares, the gripper stays vertical, the null
   space keeps the joints away from their limits, and we respect the Panda velocity limits).

All the details and results are in `report/report.pdf`.

## Notes

- Only the end-effector avoids the obstacle, the rest of the arm is not checked for collisions.

## References

- S. M. Khansari-Zadeh, A. Billard. *Learning stable nonlinear dynamical systems with Gaussian mixture models.* IEEE T-RO, 2011.
- S. M. Khansari-Zadeh, A. Billard. *A dynamical system approach to realtime obstacle avoidance.* Autonomous Robots, 2012.
- N. Figueroa, A. Billard. *Learning globally stable linear parameter varying dynamical systems.* RSS, 2018.
- L. Huber, A. Billard, J.-J. Slotine. *Avoidance of convex and concave obstacles with convergence ensured through contraction.* IEEE RA-L, 2019.
- P. Corke, J. Haviland. *Not your grandmother's toolbox: the Robotics Toolbox reinvented for Python.* ICRA, 2021.
- Lab code: `ds-opt` / `phys-gmm` (N. Figueroa, LASA-EPFL) and `dynamic_obstacle_avoidance` (L. Huber, maintained by A. Prados).

**Use of AI:** we used AI tools (Claude, by Anthropic) to help us understand some of the theory and
as support when programming and fixing errors.
