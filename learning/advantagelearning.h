#ifndef __ADVANTAGELEARNING_H__
#define __ADVANTAGELEARNING_H__

#include "abstractlearning.h"

/**
 * @brief Advantage learning as detailed in "Reinforcement Learning using LSTM"
 */
class AdvantageLearning : public AbstractLearning
{
    public:
        /**
         * @param discount_factor Discount factor used when computing cumulative rewards
         * @param learning_rate Rate at which learning occurs
         * @param kappa The smaller this factor is, the strongest the bias for
         *              better actions is.
         */
        AdvantageLearning(float discount_factor, float learning_rate, float kappa);

        virtual void actions(Episode *episode, std::vector<float> &probabilities);

    private:
        float _discount_factor;
        float _learning_rate;
        float _inv_kappa;

        // Lists so that memory does not need to be continuously reallocated
        std::vector<float> _last_values;
};

#endif