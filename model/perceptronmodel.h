#ifndef __PERCEPTRONMODEL_H__
#define __PERCEPTRONMODEL_H__

#include "nnetmodel.h"

/**
 * @brief Feed-forward neural network with a single hidden layer
 */
class PerceptronModel : public NnetModel
{
    public:
        PerceptronModel(unsigned int hidden_neurons);

        virtual Network *createNetwork(Episode *first_episode) const;

    private:
        unsigned int _hidden_neurons;
};

#endif