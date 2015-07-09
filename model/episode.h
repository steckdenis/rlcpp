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
        Episode();

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
         * @brief Number of floating-point variables in a state observation
         */
        unsigned int stateSize() const;

        /**
         * @brief Number of floating-point variables in a values tuple (number
         *        of actions)
         */
        unsigned int valueSize() const;

        /**
         * @brief Number of observations in this episode
         */
        unsigned int length() const;

        /**
         * @brief Observation for a given time step
         */
        std::vector<float> state(unsigned int t) const;

        /**
         * @brief Set of action values for a given time step
         */
        std::vector<float> values(unsigned int t) const;

        /**
         * @brief Reward at a given time step
         */
        float reward(unsigned int t) const;

        /**
         * @brief Action taken at a given time step
         */
        float action(unsigned int t) const;

    private:
        std::vector<float> _states;
        std::vector<float> _values;
        std::vector<float> _rewards;
        std::vector<int> _actions;

        unsigned int _state_size;
        unsigned int _value_size;
};

#endif