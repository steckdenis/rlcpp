/*
 * Copyright (c) 2015 Vrije Universiteit Brussel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "tablemodel.h"
#include "episode.h"

#include <algorithm>
#include <iostream>

void TableModel::swapModels()
{
    // Copy the learning table to the prediction table
    _table = _learn_table;
}

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

    // Copy the prediction table to the learning table so that learning occurs
    // on an up-do-date table
    _learn_table = _table;

    for (Episode *episode : episodes) {
        for (unsigned int t=0; t < episode->length() - 1; ++t) {
            unsigned int action = episode->action(t);

            episode->state(t, state);
            episode->values(t, values);

            // Update the value associated to the action that was taken, or
            // populate the table if this state was never encountered.
            auto it = _learn_table.find(state);

            if (it == _learn_table.end()) {
                _learn_table[state] = values;
            } else {
                it->second[action] = values[action];
            }
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
