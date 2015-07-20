#ifndef __STACKEDGRUMODEL_H__
#define __STACKEDGRUMODEL_H__

#include "recurrentnnetmodel.h"

/**
 * @brief Recurrent neural network based on Gated Recurrent Units
 *
 * This network consists of a dense layer, a GRU layer, and then another dense
 * layer. This architecture is very simple.
 */
class StackedGRUModel : public RecurrentNnetModel
{
    public:
        StackedGRUModel(unsigned int hidden_neurons);

        virtual Network *createNetwork(Episode *first_episode) const;

    private:
        unsigned int _hidden_neurons;
};

#endif