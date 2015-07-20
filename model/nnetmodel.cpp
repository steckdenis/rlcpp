#include "nnetmodel.h"
#include "episode.h"

NnetModel::NnetModel()
: _network(nullptr)
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

    // Create a big matrix with one column per input/output pair
    std::size_t total_size = 0;

    for (Episode *episode : episodes) {
        total_size += episode->length() - 1;
    }

    Eigen::MatrixXf inputs(episodes[0]->stateSize(), total_size);
    Eigen::MatrixXf outputs(episodes[0]->valueSize(), total_size);
    Eigen::MatrixXf weights(episodes[0]->valueSize(), total_size);

    // Fill this matrix
    int index = 0;

    for (Episode *episode : episodes) {
        // Create the network if needed
        if (!_network) {
            _network = createNetwork(episode);
        }

        // Learn all the values obtained during the episode
        for (unsigned int t=0; t < episode->length() - 1; ++t) {
            unsigned int action = episode->action(t);

            episode->state(t, state);
            episode->values(t, values);

            vectorToCol(state, inputs, index);
            vectorToCol(values, outputs, index);

            // Use only the value associated with the action that has been taken
            // when computing the errors
            weights.col(index).setZero();
            weights(action, index) = 1.0f;

            ++index;
        }
    }

    // Train the network on that data
    _network->train(inputs, outputs, weights, 1, 100, true);
}

void NnetModel::vectorToVector(const std::vector<float> &stl, Vector &eigen)
{
    eigen.resize(stl.size());

    for (std::size_t i=0; i<stl.size(); ++i) {
        eigen(i) = stl[i];
    }
}

void NnetModel::vectorToCol(const std::vector<float> &stl, Matrix &matrix, int col)
{
    for (std::size_t i=0; i<stl.size(); ++i) {
        matrix(i, col) = stl[i];
    }
}
