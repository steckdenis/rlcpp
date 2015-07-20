#ifndef __PARALLELGRUMODEL_H__
#define __PARALLELGRUMODEL_H__

#include "recurrentnnetmodel.h"

/**
 * @brief Recurrent neural network based on Gated Recurrent Units
 *
 * This network consists of a single-hidden-layer perceptron working in parallel
 * with a layer of GRU units. The output of the GRU is connected to the input of
 * the hidden layer, and the output of the hidden layer is connected to the input
 * of the GRU. The output of the hidden layer and the GRU are added to produce
 * the output of the network.
 *
 * This somewhat complex architecture is the one used in
 * "Reinforcement Learning with Long Short-Term Memory", Bram Bakker, 2001. The
 * LSTM units have been replaced with GRU units.
 */
class ParallelGRUModel : public RecurrentNnetModel
{
    public:
        ParallelGRUModel(unsigned int hidden_neurons);

        virtual Network *createNetwork(Episode *first_episode) const;

    private:
        unsigned int _hidden_neurons;
};

#endif