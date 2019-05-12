cmake_minimum_required(VERSION 2.8.3)
project(deep-tracking)

find_package(catkin REQUIRED COMPONENTS
    message_generation
    rospy
    sensor_msgs
    std_msgs
)

generate_messages(
    DEPENDENCIES
    sensor_msgs
    std_msgs
)

catkin_package(
    CATKIN_DEPENDS rospy sensor_msgs std_msgs message_runtime
)

include_directories(
    ${catkin_INCLUDE_DIRS}    
)
