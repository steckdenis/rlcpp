#include "nnetmodel.h"
#include "episode.h"

NnetModel::NnetModel(unsigned int hidden_neurons)
: _hidden_neurons(hidden_neurons)
{
}

void NnetModel::values(Episode *episode, std::vector<float> &rs)
{
    if (!_network) {
        // No model available, clear out rs
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Convert the last state to an Eigen vector
        Eigen::MatrixXf last_state;
        ocropus::Sequence inputs;

        episode->state(episode->length() - 1, rs);
        vectorToBatch(rs, last_state);

        inputs.push_back(last_state);

        // Feed this input to the network
        ocropus::set_inputs(_network.get(), inputs);
        _network->forward();

        // The first column of the last output is what we want
        const ocropus::Mat &last_output = _network->outputs.back();

        rs.resize(episode->valueSize());

        for (std::size_t i=0; i<rs.size(); ++i) {
            rs[i] = last_output(i, 0);
        }
    }
}

void NnetModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;
    Eigen::MatrixXf matrix_state;
    Eigen::MatrixXf matrix_values;

    for (Episode *episode : episodes) {
        Eigen::VectorXf input(episode->stateSize());

        // Create the network if needed
        if (!_network) {
            _network = ocropus::layer("Stacked", episode->stateSize(), episode->valueSize(), {}, {
                ocropus::layer("SigmoidLayer", episode->stateSize(), _hidden_neurons, {}, {}),
                ocropus::layer("LinearLayer", _hidden_neurons, episode->valueSize(), {}, {})
            });
        }

        // Make a sequence of observations and values
        ocropus::Sequence inputs;
        ocropus::Sequence outputs;

        for (unsigned int t=0; t < episode->length() - 1; ++t) {
            unsigned int action = episode->action(t);

            episode->state(t, state);
            episode->values(t, values);

            vectorToBatch(state, matrix_state);
            vectorToBatch(values, matrix_values);

            inputs.push_back(matrix_state);
            outputs.push_back(matrix_values);
        }

        // Train the network
        ocropus::train(_network.get(), inputs, outputs);
    }
}

void NnetModel::vectorToBatch(const std::vector<float> &stl, Eigen::MatrixXf &eigen)
{
    eigen.resize(stl.size(), 1);

    for (std::size_t i=0; i<stl.size(); ++i) {
        eigen(i, 0) = stl[i];
    }
}

