# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        #机器人类设置
        num_envs = 4096 #同时并行运行4096个独立机器人
        num_observations = 235 #包含机器人本体感知（关节位置/速度/力矩）、足端接触状态、IMU数据（姿态/角速度）及地形特征等信息。
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        #表示采用对称训练模式，策略网络（Policy）和值函数网络（Critic）使用相同的观测输入，而非不对称式专家知识传递。
        num_actions = 12 #对应四足机器人的12个驱动关节
        env_spacing = 3.  # not used with heightfields/trimeshes 默认环境间距，还不知道具体含义
        send_timeouts = True # send time out information to the algorithm显式通知算法回合因超时终止，避免误判为终止状态，确保时序差分误差（TD-error）正确计算。
        episode_length_s = 20 # episode length in seconds 这个参数的存在可能是训练算法的需要 还需了解

    class terrain:
        #地形类设置
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]水平方向每像素对应0.1米，控制地形细节精度，值越小地形分辨率越高
        vertical_scale = 0.005 # [m]垂直方向高度缩放比例，每单位高度数据对应5毫米
        border_size = 25 # [m]地形边界扩展25米，防止机器人跑出有效区域
        curriculum = True #启用渐进式难度提升
        static_friction = 1.0 #静摩擦系数（橡胶-混凝土接触约1.0）
        dynamic_friction = 1.0 #动摩擦系数（通常≤静摩擦）
        restitution = 0. #碰撞恢复系数（0=完全非弹性碰撞）
        # rough terrain only:
        measure_heights = True #启用地形高度测量,为策略网络提供地形先验信息
        # 在机器人周围1.6m×1.0m矩形区域布置17×11=187个探测点,通过虚拟激光雷达获取探测点高度值，构成地形高度图
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments锁定单一地形类型进行训练
        terrain_kwargs = None # Dict of arguments for selected terrain自定义地形生成参数{"slope": 0.2, "step_height": 0.1}
        max_init_terrain_level = 5 # starting curriculum state初始最大地形难度等级，用来控制课程起点
        terrain_length = 8. #单个地形块长度
        terrain_width = 8.  #单个地形块宽度
        num_rows= 10 # number of terrain rows (levels)地形难度等级数
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        #五类地形分布权重：1. 平滑斜坡2. 粗糙斜坡3. 上行楼梯4. 下行楼梯5. 离散障碍
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:坡度超过75%时强制设为垂直面
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]每10秒随机生成新指令
        heading_command = True # if true: compute ang vel command from heading error角速度由目标朝向误差计算
        # 控制命令范围
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        #初始状态
        pos = [0.0, 0.0, 1.] # x,y,z [m]位置
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]姿态
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]线速度
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]角速度
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        # 不同的控制模式
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]关节刚度kp
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]关节阻尼kd
        # action scale: target angle = actionScale * action + defaultAngle
        # 目标角度=当前角度+action*action_scale
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 # 计算负载优化4096 envs × 4 decimation = 16384并行控制线程

    class asset:
        file = "" #urdf文件路径
        name = "legged_robot"  # actor name
        # 足端刚体命名：用于接触力检测与奖励计算
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = [] # 接触惩罚部件：列表中的刚体接触将触发负奖励
        terminate_after_contacts_on = [] # 关键部件接触终止回合
        disable_gravity = False # 	禁用重力（用于特殊测试）
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixed the base of the robot固定基座（用于关节标定）
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        # 动力学参数配置
        density = 0.001 # 材料密度
        angular_damping = 0. #旋转阻尼
        linear_damping = 0. # 线性阻尼
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0. # 关节惯性 (实际电机约0.001) 模拟电机转子惯量
        thickness = 0.01 # 碰撞体壳厚度 (仅胶囊/圆柱体有效)

    class domain_rand:
        randomize_friction = True #启用地面摩擦系数随机化
        friction_range = [0.5, 1.25] # 摩擦系数均匀采样范围
        randomize_base_mass = False #基座质量扰动开关
        added_mass_range = [-1., 1.]
        push_robots = True# 启用随机外力推送
        push_interval_s = 15# 每15秒施加一次推力
        max_push_vel_xy = 1.# 推力导致的水平速度变化上限 (m/s)

    class rewards:
        class scales:
            termination = -0.0 #终止奖励，通常是负值，表示任务失败的惩罚
            tracking_lin_vel = 1.0 # 线速度追踪奖励,值越高，机器人更倾向于学习快速精确的直线运动
            tracking_ang_vel = 0.5 # 角速度追踪奖励，鼓励机器人追踪期望的角速度
            lin_vel_z = -2.0 #垂直速度惩罚，防止机器人在垂直方向乱跳。负值越大，机器人越会避免上下震荡。
            ang_vel_xy = -0.05 # 平面角速度惩罚，避免横滚和俯仰角速度过大导致失稳。
            orientation = -0. ########################
            torques = -0.00001 # 关节扭矩惩罚，限制能耗，防止机器人使用过大力矩导致损坏。
            dof_vel = -0. # 关节速度惩罚，抑制关节过快运动
            dof_acc = -2.5e-7 #关节加速度惩罚，平滑运动，减少冲击力。
            base_height = -0. #机身高度奖励
            feet_air_time =  1.0 #脚步腾空时间奖励
            collision = -1. #碰撞惩罚，避免与障碍物或自身碰撞
            feet_stumble = -0.0 # 脚步绊倒惩罚
            action_rate = -0.01 #动作变化率惩罚，避免高频控制指令导致不平滑的运动。
            stand_still = -0. # 静止奖励。若要鼓励机器人站稳，可以设成正值

        # 限制总奖励不低于 0，可以避免负奖励导致训练过早终止，但可能抑制一些探索行为。
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # 速度追踪的高斯奖励宽度。值越小，追踪精度要求越高。
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # 软约束比例，超过关节位置、速度、扭矩限制时惩罚
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.# 机身目标高度
        # 最大接触力阈值。超过此值会被惩罚，用于防止机器人过猛撞击地面。
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0# 线速度归一化系数。放大线速度的输入，让机器人更关注速度误差
            ang_vel = 0.25# 角速度归一化系数。较小的数值意味着角速度输入被压缩，可能降低角速度的灵敏度。
            dof_pos = 1.0# 关节位置归一化
            dof_vel = 0.05# 关节速度归一化。较小的系数可以抑制关节速度的剧烈变化
            height_measurements = 5.0# 高度测量归一化。放大高度输入，让机器人更敏感地感知地形变化。
        clip_observations = 100.# 观察值裁剪阈值。避免异常数据扰乱训练
        clip_actions = 100.# 动作裁剪阈值。类似地，限制动作幅度，防止策略输出极端控制指令。

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0 #参考环境索引。如果你在同时训练多个环境，比如多机器人仿真，这里决定观察哪个环境。0 表示第一个环境。
        pos = [10, 0, 6]  # [m]摄像机位置
        lookat = [11., 5, 3.]  # [m]摄像机目标点，即镜头看向的坐标。

    class sim:
        dt =  0.005
        substeps = 1 #每个仿真步只计算 1 次物理解算
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z 坐标系的上方向
        
        # PhysX 物理引擎参数
        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs解算器类型。1 表示 TGS (Tensor Gauss-Seidel)，比 PGS 更稳定，适合处理刚体和关节复杂的系统，比如四足机器人。
            num_position_iterations = 4 #位置迭代次数。影响关节和碰撞的稳定性
            num_velocity_iterations = 0 #速度迭代次数。0 表示不单独迭代速度，主要靠位置解算
            contact_offset = 0.01  # [m]接触偏移量。决定碰撞检测的提前量
            rest_offset = 0.0   # [m]静止偏移量。控制刚体接触后的最小距离
            bounce_threshold_velocity = 0.5 #0.5 [m/s]弹跳阈值速度。低于 0.5m/s 的接触不会产生弹跳
            max_depenetration_velocity = 1.0 #最大穿透修正速度。
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5# 缓冲区大小倍数。增大缓冲区可以避免接触点过多导致崩溃。
            contact_collection = 2 #接触收集策略。2 表示所有子步都收集接触信息 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0 #初始化动作噪声标准差。适度的探索噪声有助于探索新策略。
        # 网络隐藏层维度。较大的网络能学习更复杂的策略，适合复杂步态
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # 激活函数。elu 适用于连续控制，比 relu 更平滑，收敛更稳定。
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0 # 值函数损失系数，决定值函数损失在总损失中的权重。。1.0 表示策略损失和值损失等权，适合平衡策略学习和价值评估。
        use_clipped_value_loss = True #是否使用剪裁的值函数损失，PPO 本身对策略更新做了剪裁，防止策略突然大幅偏移。这里启用剪裁值损失，避免值函数估计偏差过大导致训练不稳定。
        clip_param = 0.2 #策略剪裁参数,限制策略更新幅度，防止策略走偏。
        entropy_coef = 0.01 #熵损失系数,熵奖励鼓励策略探索。值越大，策略越随机，探索更多可能性。
        num_learning_epochs = 5 #每次迭代的训练轮数。每采集一批数据，重复训练多少轮。轮数越多，数据利用率越高，但过多可能导致过拟合。
        # 小批次数量,把采集的数据分成几批处理，减少内存占用。批次越多，梯度更新越频繁。
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4学习率,控制梯度下降的步长。步长太大容易不稳定，步长太小训练很慢
        schedule = 'adaptive' # could be adaptive, fixed。学习率调度策略：adaptive：根据 KL 散度调整。训练稳定时降低学习率。fixed：固定学习率。
        gamma = 0.99 #折扣因子
        lam = 0.95 #GAE 优势估计平滑系数平衡优势估计的偏差和方差
        desired_kl = 0.01 # 目标 KL 散度，限制新旧策略的差异，避免策略变化太快
        max_grad_norm = 1. #梯度裁剪阈值，限制梯度大小，避免梯度爆炸

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration每个环境的时间步数,步数越多，采样越充分，但占用更多内存。
        max_iterations = 1500 # number of policy updates最大迭代次数。迭代次数决定训练时间。

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt