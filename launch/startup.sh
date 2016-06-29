#!/bin/bash

# Bash script to launch files necessary to run POMDP experiment.
# When shutting down experiment, close the programs in reverse order of opening.

# Exporting and sourcing to add locations of important ROS packages to path

# !!!--CHANGE THESE VARIABLES TO YOUR USERNAME AND WORKSPACE NAME--!!
user="ian"
workspace="catkin_ws"
# !!!--CHANGE THESE VARIABLES TO YOUR USERNAME AND WORKSPACE NAME--!!


export1="export ROBOT=$HOSTNAME"
source1="source /opt/ros/indigo/setup.bash"
source2="source ~/$workspace/devel/setup.bash"
export2="export ROS_PACKAGE_PATH=/home/$user/$workspace/src:/opt/ros/indigo/share:/opt/ros/indigo/stacks:/home/$user/rosbuild_ws/vicon/ros"
set_indigo="$export1 && $source1 && $source2 && $export2"

# Start roscore and vicon_sys.launch
xterm -e bash -c "$set_indigo && roscore" &
sleep 2
xterm -e bash -c "$set_indigo && \
    roscd pomdp_experiment && roslaunch ~/vicon_sys.launch" &

# Calibrate both the cop and the robber
echo "Enter ROBBER robot name (lowercase)"
read ROBBER
echo "Place robot at origin"
echo "and enter '1' to calibrate or enter '0' to skip calibration for this robot"
read calibrate_flag_robber
if [ $calibrate_flag_robber -ne 0 ] ; then
    xterm -e "$set_indigo && rosservice call \
        /vicon_bridge/calibrate_segment $ROBBER $ROBBER 0 100"
fi
echo "Enter COP robot name (lowercase)"
read COP
echo "Place robot at origin"
echo "and enter '1' to calibrate or enter '0' to skip calibration for this robot"
read calibrate_flag_cop
if [ $calibrate_flag_cop -ne 0 ] ; then
    xterm -e "$set_indigo && rosservice call \
        /vicon_bridge/calibrate_segment $COP $COP 0 100"
fi

# ssh into cop and robber and start nav stack with vicon_nav.launch
echo "ssh into robots and run the command: roslaunch cops_and_robots/launch/vicon_nav.launch"
connection_input=3
while [ $connection_input -ne 0 ]
do
    if [ $connection_input -eq 1 ] ; then
        xterm -e "ssh odroid@$COP" &
    elif [ $connection_input -eq 2 ] ; then
        xterm -e "ssh odroid@$ROBBER" &
    elif [ $connection_input -eq 3 ] ; then
        xterm -e "ssh odroid@$COP" &
        xterm -e "ssh odroid@$ROBBER" &
    fi
    echo "-------"
    echo "Enter '0' if both connections were successful"
    echo "Enter '1' to retry $COP connection"
    echo "Enter '2' to retry $ROBBER connection"
    echo "Enter '3' to retry both connections"
    read connection_input
done

# Launch ROS nodes and start running experiment
echo "When vicon_nav.launch has been started on both robots, press ENTER to run experiment"
read x
run_input=1
while [ $run_input -eq 1 ]
do
    xterm -hold -e "$set_indigo && roslaunch pomdp_experiment goals.launch cop:=$COP robber:=$ROBBER"
    echo "-------"
    echo "Enter '1' to re-run experiment"
    echo "or enter '0' to end the program"
    read run_input
done

exit 0
