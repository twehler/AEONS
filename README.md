AEONS is a simulation-experiment to simulate the evolution of biological form and behaviour, focusing on multicellularity, 
as well as its reactions to the outside environment over the generations. The simulation takes a purely Darwinistic approach, where natural selection is the only driving factor for everything that can be observed.

The life forms consist of "cells", which are simple geometric forms and exist in a voxel-based world, similar to the pop-culture video game Minecraft. Nevertheless, this is a purely scientific project. 
The practicality of a voxel-based world - apart from aestetics - is that it is easy to implement and control local environmental influences, as well as changes to this environment. Dozens of different kinds of natural rock and sediment blocks are being implemented, all of which exhibit different chemical and physical properties, like certain amounts of contained elements, a certain heat capacity when being warmed by the sun, the amount of water they can hold, as well as the soaking-speed of certain materials etc. If this project is successful, a bigger version will be constructed to also include the evolution of biochemical aspects.
The behaviour of the simulated organisms will rely on small Machine Learning algorithms in a dynamic size, which is also being dictated by natural evolution.

To ensure maximum control over the environment, the terrain is constructed from two hand-crafted images: A heightmap and a material-map. The goal is to create a complex world with realistic parameters, in which scienficially proven concepts and mechanisms about evolution can be studied in realtime.

To make sure the simulation can be run for a very long time without crashing, it is being written in Rust, with the help of the Bevy Game Engine for efficiency and the Burn crate for machine learning algorithms on the GPU.
