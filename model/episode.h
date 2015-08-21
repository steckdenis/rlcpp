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

#ifndef __EPISODE_H__
#define __EPISODE_H__

#include <vector>

/**
 * @brief List of states, actions, values and rewards.
 *
 * An episode contains the whole history of values produced during learning. The
 * world adds states in it, which are used to compute action values and rewards.
 * Then, the learning algorithm chooses an action that is also stored in the
 * episode.
 *
 * The learning algorithm can also modify the action values. After a batch of
 * episodes has been run, those values are fed to the model for learning.
 */
class Episode
{
    public:
        typedef void (*Encoder)(std::vector<float> &state);

        /**
         * @brief Constructor
         *
         * @param value_size Number of values to associate with each state (actions + other values if needed)
         * @param num_actions Number of actions possible in the world
         * @param encoder Function that encodes a state as returned by encodedState().
         *                If nullptr, an identity encoding will be used.
         */
        Episode(unsigned int value_size,
                unsigned int num_actions,
                Encoder encoder);

        /**
         * @brief Add a state to the episode.
         *
         * The length of the @p state is used to determine the state size, and
         * is required to be constant among states added to an episode
         */
        void addState(const std::vector<float> &state);

        /**
         * @brief Add a tuple of values to the episode
         *
         * The @p values list must contain one value per possible action
         */
        void addValues(const std::vector<float> &values);

        /**
         * @brief Add a reward to the episode
         */
        void addReward(float reward);

        /**
         * @brief Add an action to the episode
         */
        void addAction(int action);

        /**
         * @brief Set whether the episode has ended because the maximum time steps
         *        have been reached (instead of the goal)
         */
        void setAborted(bool aborted);

        /**
         * @brief Number of floating-point variables in an unencoded state observation
         */
        unsigned int stateSize() const;

        /**
         * @brief Number of floating-point variables in an encoded state
         */
        unsigned int encodedStateSize() const;

        /**
         * @brief Number of floating-point variables in a values tuple (may differ
         *        from the number of actions)
         */
        unsigned int valueSize() const;

        /**
         * @brief Number of actions for which values are stored
         */
        unsigned int numActions() const;

        /**
         * @brief Number of observations in this episode
         */
        unsigned int length() const;

        /**
         * @brief Whether the episode ended because the maximum number of time steps
         *        has been reached.
         *
         * If setAborted() was never called, this function returns false.
         */
        bool wasAborted() const;

        /**
         * @brief Observation for a given time step, unencoded
         */
        void state(unsigned int t, std::vector<float> &rs) const;

        /**
         * @brief Observation for a given time step, encoded using the encoder
         *        of this episode.
         */
        void encodedState(unsigned int t, std::vector<float> &rs) const;

        /**
         * @brief Set of action values for a given time step
         */
        void values(unsigned int t, std::vector<float> &rs) const;

        /**
         * @brief Update the value of an action
         */
        void updateValue(unsigned int t, unsigned int action, float value);

        /**
         * @brief Reward at a given time step
         */
        float reward(unsigned int t) const;

        /**
         * @brief Cumulative reward of this episode
         */
        float cumulativeReward() const;

        /**
         * @brief Action taken at a given time step
         */
        float action(unsigned int t) const;

    private:
        std::vector<float> _states;
        std::vector<float> _values;
        std::vector<float> _rewards;
        std::vector<int> _actions;

        Encoder _encoder;

        unsigned int _state_size;
        unsigned int _value_size;
        unsigned int _num_actions;
        bool _aborted;
};

#endif
