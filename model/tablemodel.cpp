#include "tablemodel.h"
#include "episode.h"

#include <algorithm>
#include <iostream>

void TableModel::values(Episode *episode, std::vector<float> &rs)
{
    episode->state(episode->length() - 1, rs);
    auto it = _table.find(rs);

    if (it == _table.end()) {
        // Return zeroes if nothing is stored in the table
        rs.resize(episode->valueSize());
        std::fill(rs.begin(), rs.end(), 0.0f);
    } else {
        // Return the value stored in the model
        rs = it->second;
    }
}

void TableModel::learn(const std::vector<Episode *> &episodes)
{
    std::vector<float> state;
    std::vector<float> values;

    for (Episode *episode : episodes) {
        for (unsigned int t=0; t<episode->length(); ++t) {
            episode->state(t, state);
            episode->values(t, values);

            _table[state] = values;
        }
    }
}

std::size_t TableModel::v_hash::operator()(const std::vector<float> &vector) const
{
    std::size_t acc = 0;
    auto h = std::hash<float>();

    for (float f : vector) {
        acc ^= h(f);
    }

    return acc;
}

bool TableModel::v_equal::operator()(const std::vector<float> &a, const std::vector<float> &b) const
{
    if (a.size() != b.size()) {
        return false;
    }

    return std::equal(a.begin(), a.end(), b.begin());
}