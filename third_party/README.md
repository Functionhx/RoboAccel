# Third-party projects

## ActuateX

[ActuateX](https://github.com/Functionhx/actuatex) provides reinforcement-learning
training and evaluation across Isaac Gym, Isaac Lab, and MuJoCo. It is tracked
at `third_party/actuatex` as a Git submodule pinned to a specific commit.

For an existing RoboAccel checkout, run from the repository root:

```bash
git submodule update --init --recursive third_party/actuatex
```

For a new checkout:

```bash
git clone --recurse-submodules https://github.com/Functionhx/RoboAccel.git
```

Follow ActuateX's [English guide](actuatex/README_en.md) or
[Chinese guide](actuatex/README.md) to install its simulator dependencies and
run its training and evaluation workflows. Its MIT license is preserved in
[`actuatex/LICENSE`](actuatex/LICENSE).

ActuateX currently uses the TinyMal quadruped control contract. RoboAccel's
recorded deployment uses a wheel-legged policy with different observation and
action dimensions. Adding this submodule does not connect ActuateX policies to
the existing exporter or replace the `ROBOACCEL_RL_ENV` training environment;
that requires a compatible policy adapter and separate validation. The
RoboAccel host-verification quickstart does not require this submodule.
