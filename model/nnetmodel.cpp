#include "nnetmodel.h"
#include "episode.h"

#include <nnetcpp/dense.h>
#include <nnetcpp/activation.h>

NnetModel::NnetModel(unsigned int hidden_neurons)
: _hidden_neurons(hidden_neurons),
  _network(nullptr)
{
}

NnetModel::~NnetModel()
{
    if (_network) {
        delete _network;
    }
}

void NnetModel::values(Episode *episode, std::vector<float> &rs)
{
    if (!_network) {
        // No model available, clear out rs
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Convert the last state to an Eigen vector
        Vector last_state;

        episode->state(episode->length() - 1, rs);
        vectorToVector(rs, last_state);

        // Feed this input to the network
        Vector prediction = _network->predict(last_state);

        rs.resize(episode->valueSize());

        for (std::size_t i=0; i<rs.size(); ++i) {
            rs[i] = prediction(i);
        }
    }
}

void NnetModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;

    for (Episode *episode : episodes) {
        Vector input(episode->stateSize());
        Vector output(episode->stateSize());

        // Create the network if needed
        if (!_network) {
            _network = new Network(episode->stateSize());

            Dense *dense1 = new Dense(_hidden_neurons, 0.05);
            TanhActivation *dense1_act = new TanhActivation;
            Dense *dense2 = new Dense(episode->valueSize(), 0.05);

            dense1->setInput(_network->inputPort());
            dense1_act->setInput(dense1->output());
            dense2->setInput(dense1_act->output());

            _network->addNode(dense1);
            _network->addNode(dense1_act);
            _network->addNode(dense2);
        }

        // Learn all the values obtained during the episode
        for (unsigned int t=0; t < episode->length() - 1; ++t) {
            unsigned int action = episode->action(t);

            episode->state(t, state);
            episode->values(t, values);

            vectorToVector(state, input);
            vectorToVector(values, output);

            // Perform 5 gradient steps
            for (int i=0; i<5; ++i) {
                _network->trainSample(input, output);
            }
        }
    }
}

void NnetModel::vectorToVector(const std::vector<float> &stl, Vector &eigen)
{
    eigen.resize(stl.size());

    for (std::size_t i=0; i<stl.size(); ++i) {
        eigen(i) = stl[i];
    }
}

