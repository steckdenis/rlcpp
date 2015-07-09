#include "abstractworld.h"

#include <learning/abstractlearning.h>
#include <model/abstractmodel.h>
#include <model/episode.h>

#include <iostream>
#include <cstdlib>

AbstractWorld::AbstractWorld(unsigned int num_actions)
: _num_actions(num_actions)
{
}

unsigned int AbstractWorld::numActions() const
{
    return _num_actions;
}

std::vector<Episode *> AbstractWorld::run(AbstractModel *model,
                                          AbstractLearning *learning,
                                          unsigned int num_episodes,
                                          unsigned int max_episode_length,
                                          unsigned int batch_size)
{
    std::vector<Episode *> episodes;
    std::vector<Episode *> learn_episodes;
    std::vector<float> state;
    std::vector<float> values;

    for (unsigned int e=0; e<num_episodes; ++e) {
        Episode *episode = new Episode(_num_actions);

        // Initial state
        initialState(state);
        episode->addState(state);

        // Initial value
        model->values(episode, values);
        episode->addValues(values);

        // Perform the steps
        unsigned int steps = 0;
        bool finished = false;
        float reward;

        while (steps < max_episode_length && !finished) {
            learning->actions(episode, values);

            // Choose an action according to the probabilities
            float rnd = float(std::rand() % 65536) / 65536.0f;
            float acc = 0.0f;
            float inc = 1.0f / float(episode->valueSize());
            unsigned int action;

            for (action = 0; action < episode->valueSize(); ++action) {
                acc += inc;

                if (acc > rnd) {
                    break;
                }
            }

            // Carry out the action
            step(action, finished, reward, state);

            episode->addAction(action);
            episode->addReward(reward);
            episode->addState(state);

            model->values(episode, values);
            episode->addValues(values);
        }

        // Let the learning update the values of the last state that has been visited
        learning->actions(episode, values);

        // If a batch has been finished, update the model
        episodes.push_back(episode);
        learn_episodes.push_back(episode);

        std::cout << "[Episode " << e << "] " << episode->cumulativeReward() << std::endl;

        if (learn_episodes.size() == batch_size) {
            model->learn(learn_episodes);
            learn_episodes.clear();
        }
    }

    return episodes;
}
