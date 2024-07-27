# Docker setup 

(submodule MFNLC_for_polamp_env, branch - polamp_env)

```
cd docker
sh build.sh
sh start.sh
```

# Launch Lyapunov RRT
```
sh lyapunov_rrt_experiments.bash
```


# lyapunov rrt
export PYTHONPATH=$(pwd)/MFNLC_for_polamp_env:$PYTHONPATH

