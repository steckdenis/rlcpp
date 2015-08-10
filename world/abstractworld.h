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

#ifndef __ABSTRACTWORLD_H__
#define __ABSTRACTWORLD_H__

#include <vector>

#include "model/episode.h"

class AbstractModel;
class AbstractLearning;

/**
 * @brief Provide states and rewards in response to actions
 */
class AbstractWorld
{
    public:
        AbstractWorld(unsigned int num_actions);
        virtual ~AbstractWorld() {}

        /**
         * @brief Number of actions that can be taken in this world.
         *
         * Actions are numeroted from 0 to numActions() - 1.
         */
        unsigned int numActions() const;

        /**
         * @brief Reset the environment to its initial state
         */
        virtual void reset() = 0;

        /**
         * @brief Return the initial state of this world
         *
         * This initial state is used as the first state of an episode that is
         * just started.
         */
        virtual void initialState(std::vector<float> &state) = 0;

        /**
         * @brief Execute an action in the world and observe a reward, a new state,
         *        and whether the state is a termination state.
         */
        virtual void step(unsigned int action,
                          bool &finished,
                          float &reward,
                          std::vector<float> &state) = 0;

        /**
         * @brief Execute an action for which a target state is known.
         *
         * This method is called when an initial episode is given to run(). When
         * this happens, the episode is replayed by telling the world which actions
         * were taken and what was their outcome.
         *
         * @note By default, this method simply calls step(action). Reimplement
         *       it if your world is stochastic or needs to known the target state.
         */
        virtual void stepSupervised(unsigned int action,
                                    const std::vector<float> &target_state);

        /**
         * @brief Produce a file of any format that represents the contents of
         *        the given model mapped to this world.
         *
         * @param encoder Encoder used when producing states. Can be nullptr for identity.
         */
        virtual void plotModel(AbstractModel *model, Episode::Encoder encoder);

        /**
         * @brief Run an agent in the world for a given number of episodes
         *
         * @param model Model used for learning
         * @param learning Learning algorithm
         * @param num_episodes Number of episodes run
         * @param max_episode_length Maximum number of time steps per episode
         * @param batch_size Number of episodes to run between model updates
         * @param encoder Encoder used to encode the states, can be nullptr for identity
         * @param verbose true if this method must print information about the
         *                rewards obtained by the agent. False for silent operation.
         * @param start_episode if not null, this episode is replayed before any
         *                      new episode. This allows to simulate a world from
         *                      a starting position (with history taken into account)
         *
         * @return A list of episodes. The caller must delete the episodes.
         */
        std::vector<Episode *> run(AbstractModel *model,
                                   AbstractLearning *learning,
                                   unsigned int num_episodes,
                                   unsigned int max_episode_length,
                                   unsigned int batch_size,
                                   Episode::Encoder encoder,
                                   bool verbose = true,
                                   Episode *start_episode = nullptr);

    private:
        /**
         * @brief Update _min_state and _max_state so that they contain the minimum
         *        and maximum ranges of the state variables.
         */
        void updateMinMax(const std::vector<float> &state);

    private:
        unsigned int _num_actions;
        std::vector<float> _min_state;
        std::vector<float> _max_state;
};

#endif
