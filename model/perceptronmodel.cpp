#include "perceptronmodel.h"
#include "episode.h"

#include <nnetcpp/dense.h>
#include <nnetcpp/activation.h>

PerceptronModel::PerceptronModel(unsigned int hidden_neurons)
: _hidden_neurons(hidden_neurons)
{
}

Network *PerceptronModel::createNetwork(Episode *first_episode) const
{
    Network *network = new Network(first_episode->stateSize());

    Dense *dense1 = new Dense(_hidden_neurons, 1e-4);
    TanhActivation *dense1_act = new TanhActivation;
    Dense *dense2 = new Dense(first_episode->valueSize(), 1e-4);

    dense1->setInput(network->inputPort());
    dense1_act->setInput(dense1->output());
    dense2->setInput(dense1_act->output());

    network->addNode(dense1);
    network->addNode(dense1_act);
    network->addNode(dense2);

    return network;
}
