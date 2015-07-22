#ifndef __ROSWORLD_H__
#define __ROSWORLD_H__

#include "abstractworld.h"

#include <ros/ros.h>

/**
 * @brief Produce observations and rewards from ROS subscriptions, and publish
 *        actions on ROS topics
 */
class RosWorld : public AbstractWorld
{
    public:
        /**
         * @brief Update a state variable whenever a ROS subscription is updated
         */
        struct Parser
        {
            /**
             * @brief Address of a state variable to update
             */
            float *var;
            
            /**
             * @brief Flag that has to be set to true whenever var is updated
             */
            bool *updated;
        };
        
        /**
         * @brief Publish a value on the ROS network
         */
        struct Producer
        {
            /**
             * @param values Values that this producer can publish, used to compute
             *               the number of actions of this world (every action
             *               maps to a producer and one of its value)
             */
            Producer(const std::vector<float> &values)
            : values(values)
            {}

            /**
             * @brief Publish a value
             */
            virtual void publishValue(float value) = 0;
            
        public:
            std::vector<float> values;
        };

        /**
         * @brief Simple parser for ROS message types that can be trivially
         *        converted to a float.
         */
        template<typename T>
        struct DefaultParser : public Parser
        {
            /**
             * @param node ROS node name to which to subscribe
             * @param topic Topic of the ROS node to which to subscribe
             */
            DefaultParser(const std::string &node, const std::string &topic)
            {
                ros::NodeHandle n(node);

                _subscriber = n.subscribe<T>(topic, 1, &DefaultParser<T>::update, this);
            }
            
        private:
            void update(const boost::shared_ptr<T const> &msg)
            {
                *updated = true;
                *var = float(msg->data);
            }
            
        private:
            ros::Subscriber _subscriber;
        };
        
        /**
         * @brief Simple producer for ROS messages that can be trivially
         *        produced from a float.
         */
        template<typename T>
        struct DefaultProducer : public Producer
        {
            /**
             * @param node ROS node name to which to subscribe
             * @param topic Topic of the ROS node to which to subscribe
             */
            DefaultProducer(const std::string &node,
                            const std::string &topic,
                            const std::vector<float> &values)
            : Producer(values)
            {
                ros::NodeHandle n(node);

                _publisher = n.advertise<T>(topic, 1);
            }
            
            virtual void publishValue(float value) override
            {
                T msg;

                msg.data = value;
                _publisher.publish(msg);
            }
            
        private:
            ros::Publisher _publisher;
        };

        /**
         * @brief Constructor
         * 
         * @param subscriptions List of parsers that will be used to construct
         *                      the state observations. The last subscription
         *                      will be used as a reward signal.
         * @param publications List of producers that will be used to map actions
         *                     to values published on ROS topics.
         */
        RosWorld(const std::vector<Parser *> &subscriptions,
                 const std::vector<Producer *> &publications);
        virtual ~RosWorld();
        
        virtual void initialState(std::vector<float> &state);
        virtual void reset();
        virtual void step(unsigned int action, bool &finished, float &reward, std::vector<float> &state);
        
    private:
        /**
         * @brief An action is a value published on a ROS topic
         */
        struct Action
        {
            Producer *producer;
            float value;
        };

        std::vector<Parser *> _subscriptions;
        std::vector<Producer *> _publications;
        std::vector<Action> _actions;

        std::vector<float> _state;
        bool _updated;
};

#endif