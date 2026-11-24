# dynamic_obstacle_avoidance (Lab Session 2 package, with some fixes)

This is a copy of the obstacle avoidance package from Lab Session 2 (original author Lukas Huber,
maintained by Adrián Prados, GPL-3.0, see `LICENSE`).

We only changed a few lines, all of them are marked with `# [Lab2]`:

- `visualization/animated_simulation.py` and `animated_simulation_ipython.py`: now the animation uses
  the `dynamicalSystem` argument (before it always used the linear attractor). It also accepts
  `attractorPos` as a numpy array, and we fixed a typo (`self.biSim`) in the 3D part.
- `visualization/vector_field_visualization.py`: the streamlines from `points_init` also use
  `dynamicalSystem`.

To install it, run `pip install -e .` inside this folder.
