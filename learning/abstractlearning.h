#ifndef __ABSTRACTLEARNING_H__
#define __ABSTRACTLEARNING_H__

#include <model/episode.h>

/**
 * @brief Learning algorithm
 *
 * The learning algorithm returns a probability distribution over the possible
 * actions given an episode. It can also update the action values of the episode.
 */
class AbstractLearning
{
    public:
        AbstractLearning() {}
        virtual ~AbstractLearning() {}

        virtual void actions(Episode *episode, std::vector<float> &probabilities) = 0;
};

#endif