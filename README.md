# sim
A simulator

Milestones:
1. render multiple objects in single scene (with different verticies, model matricies and textures)
2. render "movie"
3. implement drone dynamics
4. implement collision
5. parametrize drone quadrotor inputs
6. implement simple loss function, like average pixel color (gradient ascent will try to keep drone inside image frame to contribute to higher average pixel value because background is black and black is 0)
7. optimizie parameters every 1000 physic steps and restart simulation