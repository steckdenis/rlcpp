#ifndef __RECURRENTNNETMODEL_H__
#define __RECURRENTNNETMODEL_H__

#include "abstractmodel.h"

#include <nnetcpp/network.h>

/**
 * @brief Base class for recurrent neural networks.
 *
 * Recurrent neural networks are trained on input sequences, not just simple inputs.
 * This model takes care of the proper initialization and reinitialization of
 * the neural network between sequences (during training and prediction).
 */
class RecurrentNnetModel : public AbstractModel
{
    public:
        RecurrentNnetModel();
        virtual ~RecurrentNnetModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);

        /**
         * @brief Create a neural network having a number of input and output
         *        neurons adapted to @p first_episode.
         */
        virtual Network *createNetwork(Episode *first_episode) const = 0;

    private:
        Network *_network;
        unsigned int _last_episode_length;
};

#endif