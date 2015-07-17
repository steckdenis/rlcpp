#ifndef __NNETMODEL_H__
#define __NNETMODEL_H__

#include "abstractmodel.h"

#include <nnetcpp/network.h>

/**
 * @brief Perceptron model (with one hidden layer)
 */
class NnetModel : public AbstractModel
{
    public:
        NnetModel(unsigned int hidden_neurons);
        virtual ~NnetModel();

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);

    private:
        void vectorToVector(const std::vector<float> &stl, Vector &eigen);

    private:
        unsigned int _hidden_neurons;

        Network *_network;
};

#endif