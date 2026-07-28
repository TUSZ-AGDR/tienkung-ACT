# 天工行者 模仿学习 虚拟环境
本环境提供了 pro 版本机器人的sdk的部分指令，方便模仿学习测试

## ros skd
1. 所有关节的 cmd_pos 话题支持
2. 所有关节的 status 话题支持
3. 头部相机的 rgb 和 depth 话题支持

# 一、环境安装
## 软件包
```bash
 GTM
  ├── 📂 soft/                                              # 软件包
  │   ├── 📜 leisaac_v0.1.zip                               # 虚拟环境包
  │   ├── 📜 ros-humble-bodyctrl-msgs_0.0.1-1_amd64.deb     # ros2消息包
```
## 1. 安装isaccsim 与 isacclab 
参考 [安装仿真环境](https://docs.ubtrobot.com/walker-tienkung/docs/gait-training-and-deployment)

## 2. 依赖安装
```bash
cd ~/GTM/soft
sudo dpkg -i ros-humble-bodyctrl-msgs_0.0.1-1_amd64.deb
unzip leisaac_v0.1.zip
cd leisaac_v0.1/source/leisaac
conda activate env_isaaclab
pip install -e . 
pip install pyzmq

``` 

# 3. 安装ros2
``` bash 
sudo apt update && sudo apt install curl gnupg lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
sudo apt install python3-colcon-common-extensions -y
```

# 4. 运行仿真
``` bash 
conda activate env_isaaclab
cd ~/GTM/soft/leisaac_v0.1
bash start_sim.sh
```

# 5. 发布ros命令
``` bash 
conda deactivate
cd ~/GTM/soft/leisaac_v0.1
bash ros2_bridge.sh
```

# 5. 执行测试
``` bash 
conda deactivate
cd ~/GTM/soft/leisaac_v0.1/tool
bash ros_head.sh # 低头
bash ros_head.sh # 抬手
python ros_getImg.py # 保存相机采集的图片

ROS_DOMAIN_ID=146 ros2 topic list # 查看发布的所有话题
```
