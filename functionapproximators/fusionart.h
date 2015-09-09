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

#ifndef __FUSIONART_H__
#define __FUSIONART_H__

#include <Eigen/Dense>
#include <vector>

/**
 * @brief Fusion ART (Adaptive Resonance Theory) model
 *
 * Fusion ART is a model that learns a correlation between its input ports. An
 * user can put values (vectors of floats between 0 and 1) in some or all the ports,
 * setting the others to all ones. Then, the model is run, and it will adjust
 * the port values so that they become closer to a known pattern.
 *
 * For instance, using 2 ports, the first one can represent a key and the second
 * one a value. Port A is set to a key, the model is run, and port B now contains
 * the value associated with the key.
 */
class FusionART
{
    public:
        class Port {
            friend class FusionART;

            public:
                Eigen::ArrayXf value;       /*!< @brief Value of the port, input to the model or output by it */

                float weight;               /*!< @brief Weight of the port when matching a pattern (a small weight means that the port is not very important) */
                float choice;               /*!< @brief Small value between 0 and 1. The largest the value is, the less weight this port has (w = weight / choice * something) */
                float vigilence;            /*!< @brief The higher this value is (between 0 and 1), the more specific this port is (rejecting patterns too far from its value) */
                float learning_rate;        /*!< @brief Learning rate used when updating the patterns of this port */

                Port() : weight(0.0f), choice(1e-6f), vigilence(0.6f), learning_rate(0.05f)
                {}

            private:
                std::vector<Eigen::ArrayXf> patterns; /*!< @brief List of patterns, all the ports having the same number of patterns */
        };

        /**
         * @brief Add a port to this ART model
         *
         * @note FusionART does not take ownership of the port (you have to delete
         *       it yourself if it is not allocated on the stack).
         */
        void addPort(Port *port);

        /**
         * @brief Run the model, taking the values of ports and updating them
         *
         * @param learn True if the model should update its weights after this run.
         * @param pattern_index If not nullptr, address of an integer in which
         *                      the index of the best matching pattern is put.
         */
        void run(bool learn, unsigned int *pattern_index = nullptr);

        /**
         * @brief Copy all the patterns and data from another model, that must
         *        have the same number of ports.
         */
        void copyFrom(const FusionART &other);

    private:
        /**
         * @brief Return the pattern with the highest activation and a satisfied
         *        vigilence criterion.
         *
         * @return Index of the pattern, ~0 if no pattern matches.
         */
        unsigned int bestMatchingPattern(bool check_vigilence);

    private:
        struct Activation {
            unsigned int index;
            float activation;
        };

        std::vector<Port *> _ports;                     /*!< @brief List of ports of this model */
        std::vector<Activation> _pattern_activations;   /*!< @brief Activations of the different patterns */
};

#endif
