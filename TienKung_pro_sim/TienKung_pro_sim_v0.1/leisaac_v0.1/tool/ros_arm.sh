source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=146
ros2 topic pub /arm/cmd_pos bodyctrl_msgs/msg/CmdSetMotorPosition "{cmds: [{name: 12, pos: 1.5, spd: 0.2, cur: 5.0 },{name: 22, pos: -1.5, spd: 0.2, cur: 5.0 },{name: 14, pos: -0.9, spd: 0.2, cur: 5.0 },{name: 24, pos: -0.9, spd: 0.2, cur: 5.0 }]}"
