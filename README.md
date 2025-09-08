# Path-Tracer-in-MSL
A basic path tracer written in Metal Shader Language.

The path tracer is entirely branchless appart from one early exit from the stepping loop as it has no cost (regardless of if it exits the loop or not, the thread is still simulated for any additional steps) but potential gains in scenes with wide open spaces where the max depth will never be hit.
