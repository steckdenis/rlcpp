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
        virtual std::vector<float> values(Episode *episode);
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