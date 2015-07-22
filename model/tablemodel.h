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

#ifndef __TABLEMODEL_H__
#define __TABLEMODEL_H__

#include "abstractmodel.h"

#include <unordered_map>

/**
 * @brief Simple model that stores action values in a dictionary indexed by state
 *
 * This model does not store any time information and always returns the values
 * associated with the last state of an episode. The history is completely ignored.
 */
class TableModel : public AbstractModel
{
    public:
        virtual void values(Episode *episode, std::vector<float> &rs);
        virtual void learn(const std::vector<Episode *> &episodes);

    private:
        struct v_hash
        {
            std::size_t operator()(const std::vector<float> &vector) const;
        };

        struct v_equal
        {
            bool operator()(const std::vector<float> &a, const std::vector<float> &b) const;
        };

        std::unordered_map<std::vector<float>, std::vector<float>, v_hash, v_equal> _table;
};

#endif