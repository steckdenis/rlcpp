#ifndef __EGREEDYLEARNING_H__
#define __EGREEDYLEARNING_H__

#include "abstractlearning.h"

/**
 * @brief Wrapper for a learning algorithm that implements the e-Greedy action selection
 */
class EGreedyLearning : public AbstractLearning
{
    public:
        /**
         * @param learning Learning algorithm that is wrapped by this e-Greedy
         * @param epsilon Probability that an exploratory step is taken
         */
        EGreedyLearning(AbstractLearning *learning, float epsilon);
        virtual ~EGreedyLearning();

        virtual void actions(Episode *episode, std::vector<float> &probabilities);

    private:
        AbstractLearning *_learning;
        float _epsilon;
};

#endif