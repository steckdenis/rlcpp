#ifndef __QLEARNING_H__
#define __QLEARNING_H__

#include "abstractlearning.h"

/**
 * @brief Well-known Q-Learning algorithm
 */
class QLearning : public AbstractLearning
{
    public:
        /**
         * @param discount_factor Discount factor used when computing cumulative rewards
         * @param learning_rate Rate at which learning occurs
         */
        QLearning(float discount_factor, float learning_rate);

        virtual void actions(Episode *episode, std::vector<float> &probabilities);

    private:
        float _discount_factor;
        float _learning_rate;

        // Lists so that memory does not need to be continuously reallocated
        std::vector<float> _last_values;
};

#endif