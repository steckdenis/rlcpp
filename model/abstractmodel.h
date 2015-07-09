#ifndef __ABSTRACTMODEL_H__
#define __ABSTRACTMODEL_H__

#include <vector>

class Episode;

/**
 * @brief Model associating episodes (histories) to state values
 */
class AbstractModel
{
    public:
        AbstractModel() {}
        ~AbstractModel() {}

        /**
         * @brief Update the model using the action values of the @p episodes
         */
        virtual void learn(const std::vector<Episode *> &episodes) = 0;

        /**
         * @brief Return the action values corresponding to the last state of
         *        @p episode.
         */
        virtual std::vector<float> values(Episode *episode) = 0;
};

#endif