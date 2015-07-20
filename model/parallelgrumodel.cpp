#include "parallelgrumodel.h"
#include "episode.h"

#include <nnetcpp/gru.h>
#include <nnetcpp/dense.h>
#include <nnetcpp/mergesum.h>
#include <nnetcpp/activation.h>

ParallelGRUModel::ParallelGRUModel(unsigned int hidden_neurons)
: _hidden_neurons(hidden_neurons)
{
}

Network *ParallelGRUModel::createNetwork(Episode *first_episode) const
{
    static const float learning_rate = 1e-3;

    Network *net = new Network(first_episode->stateSize());
    Dense *dense_inI = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_zI = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_rI = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_inH = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_zH = new Dense(_hidden_neurons, learning_rate);
    Dense *dense_rH = new Dense(_hidden_neurons, learning_rate);
    TanhActivation *act_hidden = new TanhActivation;
    Dense *dense_hidden = new Dense(_hidden_neurons, learning_rate);
    GRU *gru = new GRU(_hidden_neurons, learning_rate);
    Dense *out_gru = new Dense(first_episode->valueSize(), learning_rate);
    Dense *out_hidden = new Dense(first_episode->valueSize(), learning_rate);
    MergeSum *out = new MergeSum;

    // Connect the input to the in, Z and R ports of GRU
    dense_inI->setInput(net->inputPort());
    dense_zI->setInput(net->inputPort());
    dense_rI->setInput(net->inputPort());

    // Connect the output of the hidden layer to the in, Z and R ports of GRU
    dense_inH->setInput(act_hidden->output());
    dense_zH->setInput(act_hidden->output());
    dense_rH->setInput(act_hidden->output());

    // input -> hidden -> output
    dense_hidden->setInput(net->inputPort());
    act_hidden->setInput(dense_hidden->output());
    out_hidden->setInput(act_hidden->output());
    out->addInput(out_hidden->output());

    // Connect the output of GRU to the output of the network
    out_gru->setInput(gru->output());
    out->addInput(out_gru->output());

    // Add the nodes. The input and the GRU layer feed the rest of the network
    net->addNode(dense_hidden);
    net->addNode(act_hidden);
    net->addNode(out_hidden);

    net->addNode(dense_inI);
    net->addNode(dense_rI);
    net->addNode(dense_zI);
    net->addNode(dense_inH);
    net->addNode(dense_rH);
    net->addNode(dense_zH);

    // All the inputs of GRU are up-to-date
    net->addNode(gru);
    net->addNode(out_gru);

    // The two outputs are computed
    net->addNode(out);

    return net;
}
