# sim
A physics simulator

1. **Collision Detection**
   - **Input:** Geometric shapes or bounding boxes.
   - **Algorithms:**
     - **Convex Hull Computation (Qhull)**: Simplifies shapes to convex hulls.
     - **Sweep and Prune (Broad-Phase)**: Sorts AABBs along axes to identify potential collisions.
     - **Bounding Volume Hierarchy (Mid-Phase BVH)**: Narrows potential collisions using hierarchies.
     - **Gilbert-Johnson-Keerthi (GJK) and Minkowski Portal Refinement (MPR) (Near-Phase)**: Exact collision detection for primitive and mesh geometries.
   - **Output:** Collision results, contact points, normals, penetration details.
   - **Purpose:** Efficiently detect and manage collisions at various phases.

2. **Dynamics and Kinematics**
   - **Input:** Joint configurations and velocities.
   - **Algorithms:**
     - **Forward and Inverse Kinematics**: Calculate positions/orientations and determine necessary joint angles.
     - **Recursive Newton-Euler (RNE)**: Computes forces for given accelerations.
     - **Composite Rigid Body (CRB)**: Computes joint-space inertia matrix.
   - **Output:** Body positions, orientations, joint torques, and inertia matrices.
   - **Purpose:** Compute movements and forces within articulated structures.

3. **Constraint Solver**
   - **Input:** Contact information, friction parameters.
   - **Algorithms:**
     - **Projected Gauss-Seidel (PGS)** and **Optimization-Based Solvers**: Calculate constraint forces.
   - **Output:** Forces ensuring non-penetration and adherence to dynamic constraints.
   - **Purpose:** Resolve interactions and maintain stability.

4. **Numerical Integration**
   - **Input:** Current state and accelerations.
   - **Algorithms:** Integration methods like semi-implicit Euler or Runge-Kutta.
   - **Output:** Updated positions and velocities.
   - **Purpose:** Advance simulation over time steps.
