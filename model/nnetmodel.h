#ifndef __NNETMODEL_H__
#define __NNETMODEL_H__

#include "abstractmodel.h"

#include <nnetcpp/network.h>

/**
 * @brief Base class for non-recurrent neural networks.
 *
 * In this network, every state-action-value tuple is considered independent from
 * any history. This hypothesis is valid when a neural network has no recurrence,
 * but recurrent networks require histories to be kept in order (use
 * RecurrentNnetModel for that).
 */
class NnetModel : public AbstractModel
{
    public:
        NnetModel();
        virtual ~NnetModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);

        /**
         * @brief Create a neural network having a number of input and output
         *        neurons adapted to @p first_episode
         */
        virtual Network *createNetwork(Episode *first_episode) const = 0;

    private:
        void vectorToVector(const std::vector<float> &stl, Vector &eigen);
        void vectorToCol(const std::vector<float> &stl, Matrix &matrix, int col);

    private:
        Network *_network;
};

#endif