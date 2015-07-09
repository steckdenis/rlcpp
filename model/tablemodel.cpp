#include "tablemodel.h"
#include "episode.h"

#include <algorithm>

std::vector<float> TableModel::values(Episode *episode)
{
    auto it = _table.find(episode->state(episode->length() - 1));

    if (it == _table.end()) {
        // Return zeroes if nothing is stored in the table
        return std::vector<float>(episode->valueSize());
    } else {
        return it->second;
    }
}

void TableModel::learn(const std::vector<Episode *> &episodes)
{
    for (Episode *episode : episodes) {
        // Store the value of the last state of each episode
        unsigned int last_t = episode->length() - 1;

        _table[episode->state(last_t)] = episode->values(last_t);
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