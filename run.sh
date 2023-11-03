# terminal commands

chmod +x ~/src/uav_sim/scripts/*.py
chmod +x ~/src/racebot_control/script/*.py
cd ~/catkin_sim
catkin_make
source devel/setup.bash
# close terminal and reopen ...

pip install apriltag

roslaunch uav_sim demo1.launch
rosrun uav_sim demo1.py

roslaunch uav_sim env.launch

rqt_image_view
rostopic echo /m3e/cmd_string