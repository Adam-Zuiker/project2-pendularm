"""arm_lib — Project 2's dynamics/control/math package: linear algebra,
expression parsing, numerical integrators, arm dynamics, and PID control.

Kept separate from the node files (arm_sim_node.py, ik_node.py, ...) so this
package never imports the broker/protocol/socket layer — it's plain,
standard-library-only numerical code, testable without a live runtime.
"""
