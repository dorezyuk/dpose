# Dpose

The [dpose_core](dpose_core) library generates the Jacobian and Hessian for a given pose based on a provided obstacles vector.
The [dpose_goal_tolerance](dpose_goal_tolerance) uses the data from `dpose_core` and implements a goal-tolerance feature as a [gpp_plugin](http://github.com/dorezyuk/gpp.git).
