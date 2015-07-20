# Reinforcement learning algorithms and experiments in C++

This repository is a port of "rlpy" in C++, for performance reasons. Python in itself is fast enough (and rlpy has been designed with speed in mind), but no fast enough neural network library is available in Python. This library is built on nnetcpp (https://github.com/steckdenis/nnetcpp), which provides very efficient neural network primitives, and allows to train recurrent neural networks on a timestep-by-timestep basis instead of whole histories at once.

As in rlpy, three main classes of objects exist in this library :

* AbstractWorld: Environment and behavior of an agent. The world defines the number of possible actions, and produces observations and rewards when actions are carried out.
* AbstractLearning: Observes states and rewards and choose actions to perform.
* AbstractModel: Stores and retrieve values. For instance, a model is used to associate Q values to (state, action) pairs. A model can be discrete or based on function approximation.

# Dependencies

Unlike rlpy, this project does not depend on any other library besides nnetcpp.

# Testing

This repository contains an "rlcpp" executable, that is compiled using those commands :

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release  # Release: important for speed, also compile nnetcpp in release mode as it provides a boost of 20x
make
```

The executable takes a number of possible arguments (see main.cpp for the list of arguments), and instantiates and connects all the objects it is asked to. Here is a small example of invocation :

```
./rlcpp gridworld oneofn perceptron advantage softmax
```

* Use the GridWorld world
* Wrap it into an OneOfNWorld, so that neural-network-based function approximators can work properly
* Use a feed-forward perceptron as model for storing state-action values
* Use the Advantage learning algorithm (the model will therefore store Advantage values)
* Action selection based on Softmax.
