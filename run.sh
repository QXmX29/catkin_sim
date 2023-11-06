# terminal commands
apt install ros-melodic-joint-state-publisher-gui ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-gmapping ros-melodic-ackermann-msgs ros-melodic-effort-controllers

chmod +x ~/catkin_sim/src/uav_sim/scripts/*.py
chmod +x ~/catkin_sim/src/racebot_control/script/*.py
cd ~/catkin_sim
catkin_make
source devel/setup.bash
# close terminal and reopen ...

pip install apriltag

roslaunch uav_sim demo1.launch
rosrun uav_sim demo1.py

# add: msg
roslaunch uav_sim env.launch
roslaunch uav_sim env.launch world:=/home/thudrone/catkin_sim/src/uav_sim/world/new2.world

rqt_image_view
rostopic echo /m3e/cmd_string