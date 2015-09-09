#include "fusionart.h"

#include <assert.h>
#include <iostream>

void FusionART::addPort(Port *port)
{
    _ports.push_back(port);
}

void FusionART::run(bool learn, unsigned int *pattern_index)
{
    // Use the inputs to compute the activations of all the patterns
    for (unsigned int i=0; i<_pattern_activations.size(); ++i) {
        _pattern_activations[i].index = i;
        _pattern_activations[i].activation = 0.0f;

        for (Port *port : _ports) {
            _pattern_activations[i].activation +=
                port->weight *
                port->value.cwiseMin(port->patterns[i]).sum() /
                (port->choice + port->patterns[i].sum());
        }
    }

    // Pattern with the highest activation and a satisfied vigilence criterion
    unsigned int best_pattern = bestMatchingPattern(learn);

    if (pattern_index) {
        *pattern_index = best_pattern;
    }

    // If learning is enabled, update the best pattern (or add a new one)
    if (learn) {
        if (best_pattern == ~0) {
            std::cout << "Creating new pattern " << _pattern_activations.size() << std::endl;

            // No existing pattern matched, create a new one
            Activation act;

            best_pattern = _pattern_activations.size();
            act.index = best_pattern;
            act.activation = 0.0f;

            _pattern_activations.push_back(act);

            for (Port *port : _ports) {
                // New pattern exactly matching the value on the port
                port->patterns.push_back(port->value);
            }
        } else {
            // Update the pattern
            for (Port *port : _ports) {
                Eigen::ArrayXf &pattern = port->patterns[best_pattern];

                pattern =
                    (1.0f - port->learning_rate) * pattern +
                    port->learning_rate * port->value.cwiseMin(pattern);
            }
        }
    }

    // Adjust the output of the ports
    if (best_pattern != ~0) {
        for (Port *port : _ports) {
            port->value = port->value.cwiseMin(port->patterns[best_pattern]);
        }
    }
}

unsigned int FusionART::bestMatchingPattern(bool check_vigilence)
{
    // Sort the pattern activations by decreasing activation so that trying one
    // pattern then another takes linear time (instead of quadratic time). This
    // operation is O(n*log n), which is also under O(n²).
    std::sort(
        _pattern_activations.begin(),
        _pattern_activations.end(),
        [](const Activation &a, const Activation &b) {
            return a.activation > b.activation;
        }
    );

    // Try to find a pattern whose vigilence criterion matches
    for (unsigned int i=0; i<_pattern_activations.size(); ++i) {
        unsigned int pattern_index = _pattern_activations[i].index;
        bool ok = true;

        if (check_vigilence) {
            // Check that the vigilence criterion is satisfied for all the ports
            for (Port *port : _ports) {
                float m = port->value.cwiseMin(port->patterns[pattern_index]).sum();

                if (m < port->vigilence * port->value.sum()) {
                    ok = false;
                    break;
                }
            }
        }

        if (ok) {
            return pattern_index;
        }
    }

    return ~0;
}

void FusionART::copyFrom(const FusionART &other)
{
    _pattern_activations = other._pattern_activations;

    for (std::size_t i=0; i<_ports.size(); ++i) {
        _ports[i]->patterns = other._ports[i]->patterns;
    }
}
