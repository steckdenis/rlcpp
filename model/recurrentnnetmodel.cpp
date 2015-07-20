#include "recurrentnnetmodel.h"
#include "nnetmodel.h"
#include "episode.h"

RecurrentNnetModel::RecurrentNnetModel()
: _network(nullptr),
  _last_episode_length(0)
{
}

RecurrentNnetModel::~RecurrentNnetModel()
{
    if (_network) {
        delete _network;
    }
}

void RecurrentNnetModel::values(Episode *episode, std::vector<float> &rs)
{
    if (!_network) {
        // No model available, clear out rs
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Reset the network if a new episode has been started
        if (episode->length() <= _last_episode_length) {
            _network->reset();
        }

        _last_episode_length = episode->length();

        // Convert the last state to an Eigen vector
        Vector last_state;

        episode->state(episode->length() - 1, rs);
        NnetModel::vectorToVector(rs, last_state);

        // Feed this input to the network
        Vector prediction = _network->predict(last_state);

        rs.resize(episode->valueSize());

        for (std::size_t i=0; i<rs.size(); ++i) {
            rs[i] = prediction(i);
        }
    }
}

void RecurrentNnetModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;
    Vector input;
    Vector output;
    Vector weights;

    // Learn all the episodes separately, because they represent sequences
    // of observations that must be kept in order
    for (int i=0; i<10; ++i) {
        for (Episode *episode : episodes) {
            // Create the network if needed
            if (!_network) {
                _network = createNetwork(episode);
            }

            // Learn all the values obtained during the episode
            weights.resize(episode->valueSize());
            weights.setZero();

            for (unsigned int t=0; t < episode->length() - 1; ++t) {
                unsigned int action = episode->action(t);

                episode->state(t, state);
                episode->values(t, values);

                NnetModel::vectorToVector(state, input);
                NnetModel::vectorToVector(values, output);

                // Use only the value associated with the action that has been taken
                // when computing the errors
                weights(action) = 1.0f;

                _network->trainSample(input, output, weights);

                weights(action) = 0.0f;
            }
        }
    }
}
