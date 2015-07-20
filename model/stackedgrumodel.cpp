#include "stackedgrumodel.h"
#include "episode.h"

#include <nnetcpp/gru.h>
#include <nnetcpp/dense.h>
#include <nnetcpp/activation.h>

StackedGRUModel::StackedGRUModel(unsigned int hidden_neurons)
: _hidden_neurons(hidden_neurons)
{
}

Network *StackedGRUModel::createNetwork(Episode *first_episode) const
{
    static const float learning_rate = 1e-3;

    Network *net = new Network(first_episode->stateSize());
    Dense *dense_in = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_z = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_r = new Dense(_hidden_neurons, learning_rate);
    GRU *gru = new GRU(_hidden_neurons, learning_rate);
    Dense *out = new Dense(first_episode->valueSize(), learning_rate);

    dense_in->setInput(net->inputPort());
    dense_z->setInput(net->inputPort());
    dense_r->setInput(net->inputPort());
    gru->addInput(dense_in->output());
    gru->addZ(dense_z->output());
    gru->addR(dense_r->output());
    out->setInput(gru->output());

    net->addNode(dense_in);
    net->addNode(dense_z);
    net->addNode(dense_r);
    net->addNode(gru);
    net->addNode(out);

    return net;
}
