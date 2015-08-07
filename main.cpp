/*
 * Copyright (c) 2015 Vrije Universiteit Brussel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "learning/qlearning.h"
#include "learning/advantagelearning.h"
#include "learning/softmaxlearning.h"
#include "learning/adaptivesoftmaxlearning.h"
#include "learning/egreedylearning.h"
#include "model/tablemodel.h"
#include "model/gaussianmixturemodel.h"
#include "model/perceptronmodel.h"
#include "model/stackedgrumodel.h"
#include "model/stackedlstmmodel.h"
#include "world/tmazeworld.h"
#include "world/gridworld.h"
#include "world/polargridworld.h"
#include "world/oneofnworld.h"
#include "world/scaleworld.h"

#ifdef ROSCPP_FOUND
    #include "world/rosworld.h"

    #include <std_msgs/Float32.h>
    #include <std_msgs/Float64.h>
#endif

#include <string>
#include <fstream>
#include <iostream>

unsigned int num_episodes = 5000;
unsigned int max_timesteps = 1000;
unsigned int hidden_neurons = 100;
unsigned int batch_size = 10;
float discount_factor = 0.9f;

int main(int argc, char **argv) {
#ifdef ROSCPP_FOUND
    // Initialize ROS
    ros::init(argc, argv, "rlcpp", ros::init_options::AnonymousName | ros::init_options::NoSigintHandler);
#endif

    AbstractWorld *world = nullptr;
    AbstractModel *model = nullptr;
    AbstractLearning *learning = nullptr;
    bool random_initial = false;

    for (int i=1; i<argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "randominitial") {
            random_initial = true;
        } else if (arg == "tmaze") {
            num_episodes = 50000;
            discount_factor = 0.98f;

            world = new TMazeWorld(8, 1000);
        } else if (arg == "gridworld" || arg == "polargridworld") {
            GridWorld::Point initial, obstacle, goal;

            initial.x = 0;
            initial.y = 2;
            obstacle.x = 5;
            obstacle.y = 2;
            goal.x = 9;
            goal.y = 2;

            if (arg == "gridworld") {
                world = new GridWorld(10, 5, initial, obstacle, goal, random_initial);
            } else if (arg == "polargridworld") {
                world = new PolarGridWorld(10, 5, initial, obstacle, goal, random_initial);
            }
        } else if (arg == "pomdp") {
            if (world == nullptr) {
                std::cerr << "Put pomdp after the world to be wrapped" << std::endl;
                return 1;
            }
            world = new ScaleWorld(world, {1.0f, 0.0f});
        } else if (arg == "oneofn") {
            if (world == nullptr) {
                std::cerr << "Put oneofn after the world to be wrapped" << std::endl;
                return 1;
            }
            world = new OneOfNWorld(world, {0, 0}, {9, 4});     // 10x4 range, okay for the gridworlds and for a short tmaze (corridor of length at most 10)
#ifdef ROSCPP_FOUND
        } else if (arg == "rospendulum") {
            batch_size = 1;

            world = new RosWorld({
                new RosWorld::DefaultParser<std_msgs::Float32>("vrep", "jointAngle"),
                new RosWorld::DefaultParser<std_msgs::Float32>("vrep", "jointVelocity"),
                new RosWorld::DefaultParser<std_msgs::Float32>("vrep", "reward")
            }, {
                new RosWorld::DefaultProducer<std_msgs::Float64>("vrep", "jointTorque", {-10.0, 0.0, 10.0})
            });
#endif
        } else if (arg == "table") {
            model = new TableModel;
        } else if (arg == "gaussian") {
            model = new GaussianMixtureModel(0.60, 0.20, 0.05);       // Tailored for the gridworld
        } else if (arg == "perceptron") {
            model = new PerceptronModel(hidden_neurons);
        } else if (arg == "stackedgru") {
            model = new StackedGRUModel(hidden_neurons);
        } else if (arg == "stackedlstm") {
            model = new StackedLSTMModel(hidden_neurons);
        } else if (arg == "qlearning") {
            learning = new QLearning(discount_factor, 0.3);
        } else if (arg == "advantage") {
            learning = new AdvantageLearning(discount_factor, 0.3, 0.5);
        } else if (arg == "softmax") {
            if (learning == nullptr) {
                std::cerr << "Put softmax after the learning algorithm to be filtered" << std::endl;
                return 1;
            }

            learning = new SoftmaxLearning(learning, 0.5);
        } else if (arg == "adaptivesoftmax") {
            if (learning == nullptr) {
                std::cerr << "Put adaptivesoftmax after the learning algorithm to be filtered" << std::endl;
                return 1;
            }

            learning = new AdaptiveSoftmaxLearning(learning, 0.2);
        } else if (arg == "egreedy") {
            if (learning == nullptr) {
                std::cerr << "Put egreedy after the learning algorithm to be filtered" << std::endl;
                return 1;
            }

            learning = new EGreedyLearning(learning, 0.2);
        }
    }

    if (world == nullptr || model == nullptr || learning == nullptr) {
        std::cerr << "You have to provide a world, a model and a learning algorithm" << std::endl;
        return 1;
    }

    // Simulate the world
    std::vector<Episode *> episodes = world->run(model, learning, num_episodes, max_timesteps, batch_size);

    // Output statistics in a file that can be plotted using gnuplot
    std::ofstream stream("rewards.dat");

    for (std::size_t e=0; e<episodes.size(); ++e) {
        stream << e << '\t' << episodes[e]->cumulativeReward() << std::endl;

        delete episodes[e];
    }

    // Plot the model
    world->plotModel(model);

    delete model;
    delete learning;
    delete world;
}
