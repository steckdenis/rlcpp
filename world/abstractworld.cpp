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

#include "abstractworld.h"

#include <learning/abstractlearning.h>
#include <model/abstractmodel.h>
#include <model/episode.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>

#include <signal.h>
#include <string.h>

/**
 * @brief Used to identify when to abort AbstractWorld::run
 */
static sig_atomic_t abort_run = 0;

/**
 * @brief Called on SIGTERM, abort AbstractWorld::run
 */
static void sigterm_handler(int)
{
    abort_run = 1;
}

AbstractWorld::AbstractWorld(unsigned int num_actions)
: _num_actions(num_actions)
{
    static bool sig_setup = false;

    if (!sig_setup) {
        // Add a handler for SIGTERM
        struct sigaction action;

        memset(&action, 0, sizeof(struct sigaction));
        action.sa_handler = sigterm_handler;

        sigaction(SIGINT, &action, nullptr);
        sigaction(SIGTERM, &action, nullptr);
        sig_setup = true;
    }
}

unsigned int AbstractWorld::numActions() const
{
    return _num_actions;
}

void AbstractWorld::stepSupervised(unsigned int action,
                                   const std::vector<float> &target_state,
                                   float reward)
{
    (void) target_state;

    bool finished;
    std::vector<float> state;

    step(action, finished, reward, state);
}

std::vector<Episode *> AbstractWorld::run(AbstractModel *model,
                                          AbstractLearning *learning,
                                          unsigned int num_episodes,
                                          unsigned int max_episode_length,
                                          unsigned int batch_size,
                                          Episode::Encoder encoder,
                                          bool verbose,
                                          Episode *start_episode)
{
    std::vector<Episode *> episodes;
    std::vector<Episode *> learn_episodes;
    std::vector<float> state;
    std::vector<float> values;

    // No abortion of the run, currently
    abort_run = 0;

    for (unsigned int e=0; e<num_episodes && !abort_run; ++e) {
        Episode *episode;

        if (!start_episode) {
            // Start a new empty episode
            episode = new Episode(learning->valueSize(_num_actions), _num_actions, encoder);

            reset();
            initialState(state);
            updateMinMax(state);

            episode->addState(state);
        } else {
            // Copy the existing episode and replay its action in the world
            episode = new Episode(*start_episode);

            reset();    // This makes the assumption that the first state of the episode is the initial state of this world.

            for (unsigned int t = 0; t < episode->length() - 1; ++t) {
                episode->state(t + 1, state);
                stepSupervised(episode->action(t), state, episode->reward(t));
            }
        }

        // Initial value
        model->nextEpisode();
        model->values(episode, values);
        episode->addValues(values);

        // Perform the steps
        unsigned int steps = 0;
        bool finished = false;
        float reward;
        float td_error;

        while (steps < max_episode_length && !finished && !abort_run) {
            learning->actions(episode, values, td_error);

            // Choose an action according to the probabilities
            float rnd = float(std::rand() % 65536) / 65536.0f;
            float acc = 0.0f;
            unsigned int action;

            for (action = 0; action < episode->valueSize() - 1; ++action) {
                acc += values[action];

                if (acc > rnd) {
                    break;
                }
            }

            // Carry out the action
            step(action, finished, reward, state);
            updateMinMax(state);

            episode->addAction(action);
            episode->addReward(reward);
            episode->addState(state);

            model->values(episode, values);
            episode->addValues(values);

            steps++;
        }

        // Let the learning update the values of the last state that has been visited
        learning->actions(episode, values, td_error);

        // Tell the episode whether it has been aborted or has reached the goal
        episode->setAborted(!finished);

        // If a batch has been finished, update the model
        episodes.push_back(episode);
        learn_episodes.push_back(episode);

        if (verbose) std::cout << "[Episode " << e << "] " << episode->cumulativeReward() << std::endl;

        if (learn_episodes.size() == batch_size) {
            if (verbose) std::cout << "Learning..." << std::flush;

            model->learn(learn_episodes);
            learn_episodes.clear();

            if (verbose) std::cout << "done" << std::endl;
        }
    }

    return episodes;
}

void AbstractWorld::plotModel(AbstractModel *model, Episode::Encoder encoder)
{
    if (_min_state.size() > 2) {
        std::cout << "Cannot plot models with more than 2 state variables" << std::endl;
        return;
    }

    // Define some variables that allow to handle 1D and 2D worlds in a generic way
    bool onedimension = (_min_state.size() == 1 || _min_state[1] == _max_state[1]);
    float min_x = _min_state[0];
    float max_x = _max_state[0];
    float dx = (max_x - min_x) * 0.01f;
    float min_y = onedimension ? 0.0f : _min_state[1];
    float max_y = onedimension ? 1.0f : _max_state[1];
    float dy = onedimension ? 1.0f : (max_y - min_y) * 0.01f;

    // Open the different output files, one for each action
    std::vector<std::ofstream *> streams;
    char filename[] = "model_0.dat";

    for (unsigned int action=0; action<_num_actions; ++action) {
        filename[6] = '0' + char(action);

        streams.push_back(new std::ofstream(filename));
    }

    // Sample the model at regular points
    std::vector<float> state(_min_state.size());
    std::vector<float> values;

    for (float y = min_y; y < max_y; y += dy) {
        for (float x = min_x; x < max_x; x += dx) {
            // Make a dummy episode
            Episode episode(numActions(), numActions(), encoder);

            state[0] = x;

            if (!onedimension) {
                state[1] = y;
            }

            episode.addState(state);

            // Query the values from the model
            model->nextEpisode();
            model->valuesForPlotting(&episode, values);

            // And print them in the output streams
            for (unsigned int action=0; action<_num_actions; ++action) {
                std::ofstream &stream = *streams[action];

                stream << x;

                if (!onedimension) {
                    stream << ' ' << y;
                }

                stream << ' ' << values[action] << '\n';
            }
        }

        // Gnuplot requires that rows are separated by a blank line
        for (auto stream : streams) {
            *stream << std::endl;
        }
    }

    // Close and delete the streams
    for (auto stream : streams) {
        stream->close();
        delete stream;
    }
}

void AbstractWorld::updateMinMax(const std::vector<float> &state)
{
    if (_min_state.size() == 0) {
        // First time the state is encountered, it is a minimum and a maximum at
        // the same time
        _min_state = state;
        _max_state = state;
    } else {
        for (std::size_t i=0; i<state.size(); ++i) {
            _min_state[i] = std::min(_min_state[i], state[i]);
            _max_state[i] = std::max(_max_state[i], state[i]);
        }
    }
}

