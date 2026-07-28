source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=146
ros2 topic pub /head/cmd_pos bodyctrl_msgs/msg/CmdSetMotorPosition "{cmds: [{name: 1, pos: 0.0, spd: 0.2, cur: 5.0 },{name: 2, pos: 0.35, spd: 0.2, cur: 5.0 },{name: 3, pos: 0.0, spd: 0.2, cur: 5.0 }]}"
