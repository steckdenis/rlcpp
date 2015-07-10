#ifndef __NNETMODEL_H__
#define __NNETMODEL_H__

#include "abstractmodel.h"
#include "functionapproximators/clstm.h"

/**
 * @brief Perceptron model (with one hidden layer)
 */
class NnetModel : public AbstractModel
{
    public:
        NnetModel(unsigned int hidden_neurons);

        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);

    private:
        void vectorToBatch(const std::vector<float> &stl, Eigen::MatrixXf &eigen);

    private:
        ocropus::Network _network;
        unsigned int _hidden_neurons;
};

#endif