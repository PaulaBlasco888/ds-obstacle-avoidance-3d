# dynamic_obstacle_avoidance (Lab Session 2 package, patched)

Copy of the obstacle-avoidance package given in Lab Session 2
(original author: Lukas Huber, maintained by Adrián Prados, GPL-3.0, see `LICENSE`).

We only changed a few lines, all marked with `# [Lab2]`:

- `visualization/animated_simulation.py` and `animated_simulation_ipython.py`: the
  animation now uses the `dynamicalSystem` argument instead of always using the linear
  attractor. It also accepts a numpy array as `attractorPos`, and a typo
  (`self.biSim`) in the 3D branch is fixed.
- `visualization/vector_field_visualization.py`: the streamlines drawn from
  `points_init` also use `dynamicalSystem`.

Install with `pip install -e .` from this folder.
