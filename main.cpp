#include <learning/qlearning.h>
#include <learning/advantagelearning.h>
#include <learning/softmaxlearning.h>
#include <model/tablemodel.h>
#include <world/gridworld.h>

#include <string>
#include <fstream>
#include <iostream>

static const unsigned int num_episodes = 1000;
static const unsigned int max_timesteps = 5000;
static const unsigned int batch_size = 10;

int main(int argc, char **argv) {
    AbstractWorld *world = nullptr;
    AbstractModel *model = nullptr;
    AbstractLearning *learning = nullptr;

    for (int i=1; i<argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "gridworld" || arg == "stochasticgridworld") {
            GridWorld::Point initial, obstacle, goal;

            initial.x = 0;
            initial.y = 2;
            obstacle.x = 5;
            obstacle.y = 2;
            goal.x = 9;
            goal.y = 2;

            world = new GridWorld(10, 5, initial, obstacle, goal, arg == "stochasticgridworld");
        } else if (arg == "table") {
            model = new TableModel;
        } else if (arg == "qlearning") {
            learning = new QLearning(0.9, 0.3);
        } else if (arg == "advantage") {
            learning = new AdvantageLearning(0.9, 0.3, 0.5);
        } else if (arg == "softmax") {
            if (learning == nullptr) {
                std::cerr << "Put softmax after the learning algorithm to be filtered" << std::endl;
                return 1;
            }

            learning = new SoftmaxLearning(learning, 0.2);
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

    delete model;
    delete learning;
    delete world;
}
