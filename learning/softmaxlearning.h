#ifndef __SOFTMAXLEARNING_H__
#define __SOFTMAXLEARNING_H__

#include "abstractlearning.h"

/**
 * @brief Wrapper for a learning algorithm that implements the Softmax action selection
 */
class SoftmaxLearning : public AbstractLearning
{
    public:
        /**
         * @param learning Learning algorithm that is wrapped by this Softmax
         * @param temperature The higher this is, the more exploration there is
         */
        SoftmaxLearning(AbstractLearning *learning, float temperature);

        virtual void actions(Episode *episode, std::vector<float> &probabilities);

    private:
        AbstractLearning *_learning;
        float _temperature;
};

#endif