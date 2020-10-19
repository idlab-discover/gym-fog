import csv
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_fog.envs.env_util import deploy_service, terminate_service, change_user_rl_variables, \
    get_min_cost_milp, get_min_cost_rl, get_number_deployed_service_instances_on_node, \
    reset_node_constraints, get_service_RL, get_service_MILP, \
    getPercentageRequests

from gym_fog.envs.linkedList import LinkedList, Node
from gym_fog.milp import ilp

from gym_fog.milp.simulation import Simulation

# General Variables defining the environment
from rl.util import plotting
from rl.util.save import save_ilp_csv

MAX_STEPS = 100
MAX_USER_DELAY = 10000
MAX_OBSERVATION_SPACE = 1000
MAX_REWARD = 25000

# User Discrete Event
EQUAL = 0
INCREASE = 1
DECREASE = 2
P_EQUAL = 0.35
P_INCREASE = 0.5
P_DECREASE = 0.15

# ACTIONS - Assuming 3 Services and 45 Nodes
ACTION_DO_NOTHING = 0
ACTION_ADD_SERVICE_1_TO_NODE_1 = 1
ACTION_ADD_SERVICE_1_TO_NODE_2 = 2
ACTION_ADD_SERVICE_1_TO_NODE_3 = 3
ACTION_ADD_SERVICE_1_TO_NODE_4 = 4
ACTION_ADD_SERVICE_1_TO_NODE_5 = 5
ACTION_ADD_SERVICE_1_TO_NODE_6 = 6
ACTION_ADD_SERVICE_1_TO_NODE_7 = 7
ACTION_ADD_SERVICE_1_TO_NODE_8 = 8
ACTION_ADD_SERVICE_1_TO_NODE_9 = 9
ACTION_ADD_SERVICE_1_TO_NODE_10 = 10
ACTION_ADD_SERVICE_1_TO_NODE_11 = 11
ACTION_ADD_SERVICE_1_TO_NODE_12 = 12
ACTION_ADD_SERVICE_1_TO_NODE_13 = 13
ACTION_ADD_SERVICE_1_TO_NODE_14 = 14
ACTION_ADD_SERVICE_1_TO_NODE_15 = 15
ACTION_ADD_SERVICE_1_TO_NODE_16 = 16
ACTION_ADD_SERVICE_1_TO_NODE_17 = 17
ACTION_ADD_SERVICE_1_TO_NODE_18 = 18
ACTION_ADD_SERVICE_1_TO_NODE_19 = 19
ACTION_ADD_SERVICE_1_TO_NODE_20 = 20
ACTION_ADD_SERVICE_1_TO_NODE_21 = 21
ACTION_ADD_SERVICE_1_TO_NODE_22 = 22
ACTION_ADD_SERVICE_1_TO_NODE_23 = 23
ACTION_ADD_SERVICE_1_TO_NODE_24 = 24
ACTION_ADD_SERVICE_1_TO_NODE_25 = 25
ACTION_ADD_SERVICE_1_TO_NODE_26 = 26
ACTION_ADD_SERVICE_1_TO_NODE_27 = 27
ACTION_ADD_SERVICE_1_TO_NODE_28 = 28
ACTION_ADD_SERVICE_1_TO_NODE_29 = 29
ACTION_ADD_SERVICE_1_TO_NODE_30 = 30
ACTION_ADD_SERVICE_1_TO_NODE_31 = 31
ACTION_ADD_SERVICE_1_TO_NODE_32 = 32
ACTION_ADD_SERVICE_1_TO_NODE_33 = 33
ACTION_ADD_SERVICE_1_TO_NODE_34 = 34
ACTION_ADD_SERVICE_1_TO_NODE_35 = 35
ACTION_ADD_SERVICE_1_TO_NODE_36 = 36
ACTION_ADD_SERVICE_1_TO_NODE_37 = 37
ACTION_ADD_SERVICE_1_TO_NODE_38 = 38
ACTION_ADD_SERVICE_1_TO_NODE_39 = 39
ACTION_ADD_SERVICE_1_TO_NODE_40 = 40
ACTION_ADD_SERVICE_1_TO_NODE_41 = 41
ACTION_ADD_SERVICE_1_TO_NODE_42 = 42
ACTION_ADD_SERVICE_1_TO_NODE_43 = 43
ACTION_ADD_SERVICE_1_TO_NODE_44 = 44
ACTION_ADD_SERVICE_1_TO_NODE_45 = 45

ACTION_ADD_SERVICE_2_TO_NODE_1 = 46
ACTION_ADD_SERVICE_2_TO_NODE_2 = 47
ACTION_ADD_SERVICE_2_TO_NODE_3 = 48
ACTION_ADD_SERVICE_2_TO_NODE_4 = 49
ACTION_ADD_SERVICE_2_TO_NODE_5 = 50
ACTION_ADD_SERVICE_2_TO_NODE_6 = 51
ACTION_ADD_SERVICE_2_TO_NODE_7 = 52
ACTION_ADD_SERVICE_2_TO_NODE_8 = 53
ACTION_ADD_SERVICE_2_TO_NODE_9 = 54
ACTION_ADD_SERVICE_2_TO_NODE_10 = 55
ACTION_ADD_SERVICE_2_TO_NODE_11 = 56
ACTION_ADD_SERVICE_2_TO_NODE_12 = 57
ACTION_ADD_SERVICE_2_TO_NODE_13 = 58
ACTION_ADD_SERVICE_2_TO_NODE_14 = 59
ACTION_ADD_SERVICE_2_TO_NODE_15 = 60
ACTION_ADD_SERVICE_2_TO_NODE_16 = 61
ACTION_ADD_SERVICE_2_TO_NODE_17 = 62
ACTION_ADD_SERVICE_2_TO_NODE_18 = 63
ACTION_ADD_SERVICE_2_TO_NODE_19 = 64
ACTION_ADD_SERVICE_2_TO_NODE_20 = 65
ACTION_ADD_SERVICE_2_TO_NODE_21 = 66
ACTION_ADD_SERVICE_2_TO_NODE_22 = 67
ACTION_ADD_SERVICE_2_TO_NODE_23 = 68
ACTION_ADD_SERVICE_2_TO_NODE_24 = 69
ACTION_ADD_SERVICE_2_TO_NODE_25 = 70
ACTION_ADD_SERVICE_2_TO_NODE_26 = 71
ACTION_ADD_SERVICE_2_TO_NODE_27 = 72
ACTION_ADD_SERVICE_2_TO_NODE_28 = 73
ACTION_ADD_SERVICE_2_TO_NODE_29 = 74
ACTION_ADD_SERVICE_2_TO_NODE_30 = 75
ACTION_ADD_SERVICE_2_TO_NODE_31 = 76
ACTION_ADD_SERVICE_2_TO_NODE_32 = 77
ACTION_ADD_SERVICE_2_TO_NODE_33 = 78
ACTION_ADD_SERVICE_2_TO_NODE_34 = 79
ACTION_ADD_SERVICE_2_TO_NODE_35 = 80
ACTION_ADD_SERVICE_2_TO_NODE_36 = 81
ACTION_ADD_SERVICE_2_TO_NODE_37 = 82
ACTION_ADD_SERVICE_2_TO_NODE_38 = 83
ACTION_ADD_SERVICE_2_TO_NODE_39 = 84
ACTION_ADD_SERVICE_2_TO_NODE_40 = 85
ACTION_ADD_SERVICE_2_TO_NODE_41 = 86
ACTION_ADD_SERVICE_2_TO_NODE_42 = 87
ACTION_ADD_SERVICE_2_TO_NODE_43 = 88
ACTION_ADD_SERVICE_2_TO_NODE_44 = 89
ACTION_ADD_SERVICE_2_TO_NODE_45 = 90

ACTION_ADD_SERVICE_3_TO_NODE_1 = 91
ACTION_ADD_SERVICE_3_TO_NODE_2 = 92
ACTION_ADD_SERVICE_3_TO_NODE_3 = 93
ACTION_ADD_SERVICE_3_TO_NODE_4 = 94
ACTION_ADD_SERVICE_3_TO_NODE_5 = 95
ACTION_ADD_SERVICE_3_TO_NODE_6 = 96
ACTION_ADD_SERVICE_3_TO_NODE_7 = 97
ACTION_ADD_SERVICE_3_TO_NODE_8 = 98
ACTION_ADD_SERVICE_3_TO_NODE_9 = 99
ACTION_ADD_SERVICE_3_TO_NODE_10 = 100
ACTION_ADD_SERVICE_3_TO_NODE_11 = 101
ACTION_ADD_SERVICE_3_TO_NODE_12 = 102
ACTION_ADD_SERVICE_3_TO_NODE_13 = 103
ACTION_ADD_SERVICE_3_TO_NODE_14 = 104
ACTION_ADD_SERVICE_3_TO_NODE_15 = 105
ACTION_ADD_SERVICE_3_TO_NODE_16 = 106
ACTION_ADD_SERVICE_3_TO_NODE_17 = 107
ACTION_ADD_SERVICE_3_TO_NODE_18 = 108
ACTION_ADD_SERVICE_3_TO_NODE_19 = 109
ACTION_ADD_SERVICE_3_TO_NODE_20 = 110
ACTION_ADD_SERVICE_3_TO_NODE_21 = 111
ACTION_ADD_SERVICE_3_TO_NODE_22 = 112
ACTION_ADD_SERVICE_3_TO_NODE_23 = 113
ACTION_ADD_SERVICE_3_TO_NODE_24 = 114
ACTION_ADD_SERVICE_3_TO_NODE_25 = 115
ACTION_ADD_SERVICE_3_TO_NODE_26 = 116
ACTION_ADD_SERVICE_3_TO_NODE_27 = 117
ACTION_ADD_SERVICE_3_TO_NODE_28 = 118
ACTION_ADD_SERVICE_3_TO_NODE_29 = 119
ACTION_ADD_SERVICE_3_TO_NODE_30 = 120
ACTION_ADD_SERVICE_3_TO_NODE_31 = 121
ACTION_ADD_SERVICE_3_TO_NODE_32 = 122
ACTION_ADD_SERVICE_3_TO_NODE_33 = 123
ACTION_ADD_SERVICE_3_TO_NODE_34 = 124
ACTION_ADD_SERVICE_3_TO_NODE_35 = 125
ACTION_ADD_SERVICE_3_TO_NODE_36 = 126
ACTION_ADD_SERVICE_3_TO_NODE_37 = 127
ACTION_ADD_SERVICE_3_TO_NODE_38 = 128
ACTION_ADD_SERVICE_3_TO_NODE_39 = 129
ACTION_ADD_SERVICE_3_TO_NODE_40 = 130
ACTION_ADD_SERVICE_3_TO_NODE_41 = 131
ACTION_ADD_SERVICE_3_TO_NODE_42 = 132
ACTION_ADD_SERVICE_3_TO_NODE_43 = 133
ACTION_ADD_SERVICE_3_TO_NODE_44 = 134
ACTION_ADD_SERVICE_3_TO_NODE_45 = 135

ACTION_TERMINATE_SERVICE_1_FROM_NODE_1 = 136
ACTION_TERMINATE_SERVICE_1_FROM_NODE_2 = 137
ACTION_TERMINATE_SERVICE_1_FROM_NODE_3 = 138
ACTION_TERMINATE_SERVICE_1_FROM_NODE_4 = 139
ACTION_TERMINATE_SERVICE_1_FROM_NODE_5 = 140
ACTION_TERMINATE_SERVICE_1_FROM_NODE_6 = 141
ACTION_TERMINATE_SERVICE_1_FROM_NODE_7 = 142
ACTION_TERMINATE_SERVICE_1_FROM_NODE_8 = 143
ACTION_TERMINATE_SERVICE_1_FROM_NODE_9 = 144
ACTION_TERMINATE_SERVICE_1_FROM_NODE_10 = 145
ACTION_TERMINATE_SERVICE_1_FROM_NODE_11 = 146
ACTION_TERMINATE_SERVICE_1_FROM_NODE_12 = 147
ACTION_TERMINATE_SERVICE_1_FROM_NODE_13 = 148
ACTION_TERMINATE_SERVICE_1_FROM_NODE_14 = 149
ACTION_TERMINATE_SERVICE_1_FROM_NODE_15 = 150
ACTION_TERMINATE_SERVICE_1_FROM_NODE_16 = 136
ACTION_TERMINATE_SERVICE_1_FROM_NODE_17 = 137
ACTION_TERMINATE_SERVICE_1_FROM_NODE_18 = 138
ACTION_TERMINATE_SERVICE_1_FROM_NODE_19 = 139
ACTION_TERMINATE_SERVICE_1_FROM_NODE_20 = 140
ACTION_TERMINATE_SERVICE_1_FROM_NODE_21 = 141
ACTION_TERMINATE_SERVICE_1_FROM_NODE_22 = 142
ACTION_TERMINATE_SERVICE_1_FROM_NODE_23 = 143
ACTION_TERMINATE_SERVICE_1_FROM_NODE_24 = 144
ACTION_TERMINATE_SERVICE_1_FROM_NODE_25 = 145
ACTION_TERMINATE_SERVICE_1_FROM_NODE_26 = 146
ACTION_TERMINATE_SERVICE_1_FROM_NODE_27 = 147
ACTION_TERMINATE_SERVICE_1_FROM_NODE_28 = 148
ACTION_TERMINATE_SERVICE_1_FROM_NODE_29 = 149
ACTION_TERMINATE_SERVICE_1_FROM_NODE_30 = 150
ACTION_TERMINATE_SERVICE_1_FROM_NODE_31 = 136
ACTION_TERMINATE_SERVICE_1_FROM_NODE_32 = 137
ACTION_TERMINATE_SERVICE_1_FROM_NODE_33 = 138
ACTION_TERMINATE_SERVICE_1_FROM_NODE_34 = 139
ACTION_TERMINATE_SERVICE_1_FROM_NODE_35 = 140
ACTION_TERMINATE_SERVICE_1_FROM_NODE_36 = 141
ACTION_TERMINATE_SERVICE_1_FROM_NODE_37 = 142
ACTION_TERMINATE_SERVICE_1_FROM_NODE_38 = 143
ACTION_TERMINATE_SERVICE_1_FROM_NODE_39 = 144
ACTION_TERMINATE_SERVICE_1_FROM_NODE_40 = 145
ACTION_TERMINATE_SERVICE_1_FROM_NODE_41 = 146
ACTION_TERMINATE_SERVICE_1_FROM_NODE_42 = 147
ACTION_TERMINATE_SERVICE_1_FROM_NODE_43 = 148
ACTION_TERMINATE_SERVICE_1_FROM_NODE_44 = 149
ACTION_TERMINATE_SERVICE_1_FROM_NODE_45 = 150

ACTION_TERMINATE_SERVICE_2_FROM_NODE_1 = 151
ACTION_TERMINATE_SERVICE_2_FROM_NODE_2 = 152
ACTION_TERMINATE_SERVICE_2_FROM_NODE_3 = 153
ACTION_TERMINATE_SERVICE_2_FROM_NODE_4 = 154
ACTION_TERMINATE_SERVICE_2_FROM_NODE_5 = 155
ACTION_TERMINATE_SERVICE_2_FROM_NODE_6 = 156
ACTION_TERMINATE_SERVICE_2_FROM_NODE_7 = 157
ACTION_TERMINATE_SERVICE_2_FROM_NODE_8 = 158
ACTION_TERMINATE_SERVICE_2_FROM_NODE_9 = 159
ACTION_TERMINATE_SERVICE_2_FROM_NODE_10 = 160
ACTION_TERMINATE_SERVICE_2_FROM_NODE_11 = 161
ACTION_TERMINATE_SERVICE_2_FROM_NODE_12 = 162
ACTION_TERMINATE_SERVICE_2_FROM_NODE_13 = 163
ACTION_TERMINATE_SERVICE_2_FROM_NODE_14 = 164
ACTION_TERMINATE_SERVICE_2_FROM_NODE_15 = 165
ACTION_TERMINATE_SERVICE_2_FROM_NODE_16 = 166
ACTION_TERMINATE_SERVICE_2_FROM_NODE_17 = 167
ACTION_TERMINATE_SERVICE_2_FROM_NODE_18 = 168
ACTION_TERMINATE_SERVICE_2_FROM_NODE_19 = 169
ACTION_TERMINATE_SERVICE_2_FROM_NODE_20 = 170
ACTION_TERMINATE_SERVICE_2_FROM_NODE_21 = 171
ACTION_TERMINATE_SERVICE_2_FROM_NODE_22 = 172
ACTION_TERMINATE_SERVICE_2_FROM_NODE_23 = 173
ACTION_TERMINATE_SERVICE_2_FROM_NODE_24 = 174
ACTION_TERMINATE_SERVICE_2_FROM_NODE_25 = 175
ACTION_TERMINATE_SERVICE_2_FROM_NODE_26 = 176
ACTION_TERMINATE_SERVICE_2_FROM_NODE_27 = 177
ACTION_TERMINATE_SERVICE_2_FROM_NODE_28 = 178
ACTION_TERMINATE_SERVICE_2_FROM_NODE_29 = 179
ACTION_TERMINATE_SERVICE_2_FROM_NODE_30 = 180
ACTION_TERMINATE_SERVICE_2_FROM_NODE_31 = 181
ACTION_TERMINATE_SERVICE_2_FROM_NODE_32 = 182
ACTION_TERMINATE_SERVICE_2_FROM_NODE_33 = 183
ACTION_TERMINATE_SERVICE_2_FROM_NODE_34 = 184
ACTION_TERMINATE_SERVICE_2_FROM_NODE_35 = 185
ACTION_TERMINATE_SERVICE_2_FROM_NODE_36 = 186
ACTION_TERMINATE_SERVICE_2_FROM_NODE_37 = 187
ACTION_TERMINATE_SERVICE_2_FROM_NODE_38 = 188
ACTION_TERMINATE_SERVICE_2_FROM_NODE_39 = 189
ACTION_TERMINATE_SERVICE_2_FROM_NODE_40 = 190
ACTION_TERMINATE_SERVICE_2_FROM_NODE_41 = 191
ACTION_TERMINATE_SERVICE_2_FROM_NODE_42 = 192
ACTION_TERMINATE_SERVICE_2_FROM_NODE_43 = 193
ACTION_TERMINATE_SERVICE_2_FROM_NODE_44 = 194
ACTION_TERMINATE_SERVICE_2_FROM_NODE_45 = 195

ACTION_TERMINATE_SERVICE_3_FROM_NODE_1 = 196
ACTION_TERMINATE_SERVICE_3_FROM_NODE_2 = 197
ACTION_TERMINATE_SERVICE_3_FROM_NODE_3 = 198
ACTION_TERMINATE_SERVICE_3_FROM_NODE_4 = 199
ACTION_TERMINATE_SERVICE_3_FROM_NODE_5 = 200
ACTION_TERMINATE_SERVICE_3_FROM_NODE_6 = 201
ACTION_TERMINATE_SERVICE_3_FROM_NODE_7 = 202
ACTION_TERMINATE_SERVICE_3_FROM_NODE_8 = 203
ACTION_TERMINATE_SERVICE_3_FROM_NODE_9 = 204
ACTION_TERMINATE_SERVICE_3_FROM_NODE_10 = 205
ACTION_TERMINATE_SERVICE_3_FROM_NODE_11 = 206
ACTION_TERMINATE_SERVICE_3_FROM_NODE_12 = 207
ACTION_TERMINATE_SERVICE_3_FROM_NODE_13 = 208
ACTION_TERMINATE_SERVICE_3_FROM_NODE_14 = 209
ACTION_TERMINATE_SERVICE_3_FROM_NODE_15 = 210
ACTION_TERMINATE_SERVICE_3_FROM_NODE_16 = 211
ACTION_TERMINATE_SERVICE_3_FROM_NODE_17 = 212
ACTION_TERMINATE_SERVICE_3_FROM_NODE_18 = 213
ACTION_TERMINATE_SERVICE_3_FROM_NODE_19 = 214
ACTION_TERMINATE_SERVICE_3_FROM_NODE_20 = 215
ACTION_TERMINATE_SERVICE_3_FROM_NODE_21 = 216
ACTION_TERMINATE_SERVICE_3_FROM_NODE_22 = 217
ACTION_TERMINATE_SERVICE_3_FROM_NODE_23 = 218
ACTION_TERMINATE_SERVICE_3_FROM_NODE_24 = 219
ACTION_TERMINATE_SERVICE_3_FROM_NODE_25 = 220
ACTION_TERMINATE_SERVICE_3_FROM_NODE_26 = 221
ACTION_TERMINATE_SERVICE_3_FROM_NODE_27 = 222
ACTION_TERMINATE_SERVICE_3_FROM_NODE_28 = 223
ACTION_TERMINATE_SERVICE_3_FROM_NODE_29 = 224
ACTION_TERMINATE_SERVICE_3_FROM_NODE_30 = 225
ACTION_TERMINATE_SERVICE_3_FROM_NODE_31 = 226
ACTION_TERMINATE_SERVICE_3_FROM_NODE_32 = 227
ACTION_TERMINATE_SERVICE_3_FROM_NODE_33 = 228
ACTION_TERMINATE_SERVICE_3_FROM_NODE_34 = 229
ACTION_TERMINATE_SERVICE_3_FROM_NODE_35 = 230
ACTION_TERMINATE_SERVICE_3_FROM_NODE_36 = 231
ACTION_TERMINATE_SERVICE_3_FROM_NODE_37 = 232
ACTION_TERMINATE_SERVICE_3_FROM_NODE_38 = 233
ACTION_TERMINATE_SERVICE_3_FROM_NODE_39 = 234
ACTION_TERMINATE_SERVICE_3_FROM_NODE_40 = 235
ACTION_TERMINATE_SERVICE_3_FROM_NODE_41 = 236
ACTION_TERMINATE_SERVICE_3_FROM_NODE_42 = 237
ACTION_TERMINATE_SERVICE_3_FROM_NODE_43 = 238
ACTION_TERMINATE_SERVICE_3_FROM_NODE_44 = 239
ACTION_TERMINATE_SERVICE_3_FROM_NODE_45 = 240


def getCostReward(cost_rl, cost_milp):
    # 4) MIN Cost
    # Up to 10% Higher Cost
    if cost_milp <= cost_rl <= 1.10 * cost_milp:
        return 10
    elif 1.10 * cost_milp < cost_rl <= 1.25 * cost_milp:
        return -10
    elif 1.25 * cost_milp < cost_rl <= 1.75 * cost_milp:
        return -25
    elif 1.75 * cost_milp < cost_rl <= 2 * cost_milp:
        return -50
    elif 2 * cost_milp < cost_rl <= 3 * cost_milp:
        return -75
    elif 3 * cost_milp < cost_rl <= 4 * cost_milp:
        return -90
    else:  # >4
        return -100


def getServiceReward(RL, MILP):
    if RL == MILP:
        return 5
    else:
        return 0


class FogEnvEnergyEfficiencyLarge(gym.Env):
    """A Fog Computing environment based on OpenAI gym"""

    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, name, number_users, dynamic=False, full_dynamic=False):
        # Define action and observation space
        # They must be gym.spaces objects

        super(FogEnvEnergyEfficiencyLarge, self).__init__()

        self.name = name
        self.dynamic = dynamic
        self.full_dynamic = full_dynamic
        self.__version__ = "0.0.1"
        self.seed()

        logging.info("Fog Env Energy Efficiency Large - Version {}".format(self.__version__))

        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n
        self.num_actions = 241
        self.action_space = spaces.Discrete(self.num_actions)

        # Create MILP Simulation
        self.initial_users = number_users
        self.number_users = number_users
        self.initial_run = 1
        self.user_state = EQUAL
        self.simulation = Simulation("CPLEX_JOURNAL_JNCA", self.number_users, self.initial_run, "MIN_COST", "NONE",
                                     "NONE", "NONE", "NONE", "NONE")

        self.plot_ilp_time = np.array([0])

        # Observations: 100 Metrics!!
        # User Requests
        # % accepted requests
        # S1 RL
        # S2 RL
        # S3 RL
        # S1 MILP
        # S2 MILP
        # S3 MILP
        # Cost RL                      -> Compare with MILP solution
        # Cost MILP                    -> Compare with MILP solution - 10 metrics
        # "Flag of S1 deployed in node 1" ... Flag of S3 deployed in node 45": 135 Observation Metrics
        # N1 CPU, N1 RAM, N1 Band ... N45 CPU, N45 RAM, N45 Band: 135 Observation Metrics
        # Higher Observation Space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, #100
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, #200
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          ]),
            high=np.array([self.simulation.MAX_USERS,  # User Requests
                           100,  # % accepted requests
                           self.simulation.MAX_REPLICAS,  # S1 RL
                           self.simulation.MAX_REPLICAS,  # S2 RL
                           self.simulation.MAX_REPLICAS,  # S3 RL
                           self.simulation.MAX_REPLICAS,  # S1 MILP
                           self.simulation.MAX_REPLICAS,  # S2 MILP
                           self.simulation.MAX_REPLICAS,  # S3 MILP
                           MAX_OBSERVATION_SPACE,  # Cost RL
                           MAX_OBSERVATION_SPACE,  # Cost MILP
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0,
                           self.simulation.node_cpu[0], self.simulation.node_mem[0], self.simulation.node_band[0],
                           self.simulation.node_cpu[1], self.simulation.node_mem[1], self.simulation.node_band[1],
                           self.simulation.node_cpu[2], self.simulation.node_mem[2], self.simulation.node_band[2],
                           self.simulation.node_cpu[3], self.simulation.node_mem[3], self.simulation.node_band[3],
                           self.simulation.node_cpu[4], self.simulation.node_mem[4], self.simulation.node_band[4],
                           self.simulation.node_cpu[5], self.simulation.node_mem[5], self.simulation.node_band[5],
                           self.simulation.node_cpu[6], self.simulation.node_mem[6], self.simulation.node_band[6],
                           self.simulation.node_cpu[7], self.simulation.node_mem[7], self.simulation.node_band[7],
                           self.simulation.node_cpu[8], self.simulation.node_mem[8], self.simulation.node_band[8],
                           self.simulation.node_cpu[9], self.simulation.node_mem[9], self.simulation.node_band[9],
                           self.simulation.node_cpu[10], self.simulation.node_mem[10], self.simulation.node_band[10],
                           self.simulation.node_cpu[11], self.simulation.node_mem[11], self.simulation.node_band[11],
                           self.simulation.node_cpu[12], self.simulation.node_mem[12], self.simulation.node_band[12],
                           self.simulation.node_cpu[13], self.simulation.node_mem[13], self.simulation.node_band[13],
                           self.simulation.node_cpu[14], self.simulation.node_mem[14], self.simulation.node_band[14],
                           self.simulation.node_cpu[15], self.simulation.node_mem[15], self.simulation.node_band[15],
                           self.simulation.node_cpu[16], self.simulation.node_mem[16], self.simulation.node_band[16],
                           self.simulation.node_cpu[17], self.simulation.node_mem[17], self.simulation.node_band[17],
                           self.simulation.node_cpu[18], self.simulation.node_mem[18], self.simulation.node_band[18],
                           self.simulation.node_cpu[19], self.simulation.node_mem[19], self.simulation.node_band[19],
                           self.simulation.node_cpu[20], self.simulation.node_mem[20], self.simulation.node_band[20],
                           self.simulation.node_cpu[21], self.simulation.node_mem[21], self.simulation.node_band[21],
                           self.simulation.node_cpu[22], self.simulation.node_mem[22], self.simulation.node_band[22],
                           self.simulation.node_cpu[23], self.simulation.node_mem[23], self.simulation.node_band[23],
                           self.simulation.node_cpu[24], self.simulation.node_mem[24], self.simulation.node_band[24],
                           self.simulation.node_cpu[25], self.simulation.node_mem[25], self.simulation.node_band[25],
                           self.simulation.node_cpu[26], self.simulation.node_mem[26], self.simulation.node_band[26],
                           self.simulation.node_cpu[27], self.simulation.node_mem[27], self.simulation.node_band[27],
                           self.simulation.node_cpu[28], self.simulation.node_mem[28], self.simulation.node_band[28],
                           self.simulation.node_cpu[29], self.simulation.node_mem[29], self.simulation.node_band[29],
                           self.simulation.node_cpu[30], self.simulation.node_mem[30], self.simulation.node_band[30],
                           self.simulation.node_cpu[31], self.simulation.node_mem[31], self.simulation.node_band[31],
                           self.simulation.node_cpu[32], self.simulation.node_mem[32], self.simulation.node_band[32],
                           self.simulation.node_cpu[33], self.simulation.node_mem[33], self.simulation.node_band[33],
                           self.simulation.node_cpu[34], self.simulation.node_mem[34], self.simulation.node_band[34],
                           self.simulation.node_cpu[35], self.simulation.node_mem[35], self.simulation.node_band[35],
                           self.simulation.node_cpu[36], self.simulation.node_mem[36], self.simulation.node_band[36],
                           self.simulation.node_cpu[37], self.simulation.node_mem[37], self.simulation.node_band[37],
                           self.simulation.node_cpu[38], self.simulation.node_mem[38], self.simulation.node_band[38],
                           self.simulation.node_cpu[39], self.simulation.node_mem[39], self.simulation.node_band[39],
                           self.simulation.node_cpu[40], self.simulation.node_mem[40], self.simulation.node_band[40],
                           self.simulation.node_cpu[41], self.simulation.node_mem[41], self.simulation.node_band[41],
                           self.simulation.node_cpu[42], self.simulation.node_mem[42], self.simulation.node_band[42],
                           self.simulation.node_cpu[43], self.simulation.node_mem[43], self.simulation.node_band[43],
                           self.simulation.node_cpu[44], self.simulation.node_mem[44], self.simulation.node_band[44],
                           ]),
            dtype=np.float32
        )

        # MILP Execution
        # Start with the empty list
        self.list = LinkedList()

        # Execute first MILP calculation
        ilp.run(self.simulation)

        # Add first simulation
        self.list.head = Node(self.simulation, number_users)

        self.plot_ilp_time = np.append(self.plot_ilp_time,
                                       self.simulation.duration)  # save duration of first ILP calculation

        if dynamic:
            logging.info("Dynamic Case: Calculate all ILP formulations before running Agent...")
            for i in range(self.simulation.MAX_USERS + 1):
                if i > 1:
                    logging.info("Add sim: " + str(i) + " | Number_users: " + (str(i)))
                    sim = Simulation("CPLEX_JOURNAL_JNCA", i, self.initial_run, "MIN_COST", "NONE", "NONE", "NONE",
                                     "NONE", "NONE")

                    # Calculate MILP
                    ilp.run(sim)

                    # Save ILP execution time
                    self.plot_ilp_time = np.append(self.plot_ilp_time, sim.duration)

                    # Add Simulation to the List
                    self.list.add(sim, i)

            logging.info("Simulation List size: " + str(self.list.get_size()))

            # Save ILP execution Time in CSV
            save_ilp_csv(self.simulation.MAX_USERS, self.plot_ilp_time)

            # Save graph
            stats = plotting.ILP_Stats(execution_time=self.plot_ilp_time)
            plotting.plot_ilp_stats("ilp", stats)

        # Decision Variables
        self.max_user_delay = MAX_USER_DELAY
        self.max_reward = MAX_REWARD
        self.rl_previous_placement = np.zeros((self.simulation.NUM_APPLICATIONS, self.simulation.MAX_IDS,
                                               self.simulation.NUM_SERVICES, self.simulation.MAX_REPLICAS,
                                               self.simulation.NUM_NODES))

        self.rl_placement_matrix = np.zeros((self.simulation.NUM_APPLICATIONS, self.simulation.MAX_IDS,
                                             self.simulation.NUM_SERVICES, self.simulation.MAX_REPLICAS,
                                             self.simulation.NUM_NODES))

        self.rl_user_delay = np.zeros(self.simulation.NUM_USER_REQUESTS)

        self.rl_node_av_cpu = np.zeros(self.simulation.NUM_NODES)
        self.rl_node_av_mem = np.zeros(self.simulation.NUM_NODES)
        self.rl_node_av_band = np.zeros(self.simulation.NUM_NODES)

        reset_node_constraints(self)

        # fill with high values
        for u in range(self.simulation.NUM_USER_REQUESTS):
            self.rl_user_delay[u] = MAX_USER_DELAY

        self.rl_user_service_association = np.zeros(
            (self.simulation.NUM_USER_REQUESTS, self.simulation.NUM_APPLICATIONS,
             self.simulation.MAX_IDS, self.simulation.NUM_SERVICES,
             self.simulation.MAX_REPLICAS, self.simulation.NUM_NODES))

        # Info
        self.total_reward = None

        # episode over
        self.episode_over = False
        self.info = {}

        # Keywords for Reward calculation
        self.constraint_max_services_on_node = False
        self.constraint_add_service_max_replicas_reached = False
        self.constraint_terminate_service_without_deployed_services = False
        self.constraint_node_capacity = False
        self.deploy_service = False
        self.terminate_service = False

        # Execute MILP
        # logging.info("Reset mode: MILP running...")
        # ilp.run(self.simulation)

    def step(self, action):
        # Execute one time step within the environment
        self.take_action(action)

        reward = self.get_reward()
        self.total_reward += reward

        # Print Step and Total Reward
        logging.info(
            f"Step: {int(self.current_step)} | "
            f"Reward: {int(reward)} | "
            f"Total Reward: {int(self.total_reward)} | "
        )

        # Change User Requests at a 5 step interval for dynamic use case
        if self.dynamic:
            if self.full_dynamic:
                if self.current_step % 5 == 0:
                    self.user_request_management()
            else:
                if self.current_step % 2 == 0:
                    self.user_request_management()

        ob = self.get_state()

        self.info = dict(
            total_reward=self.total_reward,
        )

        # Update Reward Keywords
        self.constraint_max_services_on_node = False
        self.constraint_add_service_max_replicas_reached = False
        self.constraint_terminate_service_without_deployed_services = False
        self.terminate_service = False
        self.constraint_node_capacity = False

        # return ob, reward, self.episode_over, self.info
        return np.array(ob), reward, self.episode_over, self.info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0

        self.initial_users = self.initial_users
        self.number_users = self.initial_users
        self.initial_run = 1
        self.user_state = EQUAL

        if self.dynamic:
            self.simulation = self.list.search(self.number_users).get_sim()

        self.rl_previous_placement = np.zeros((self.simulation.NUM_APPLICATIONS, self.simulation.MAX_IDS,
                                               self.simulation.NUM_SERVICES, self.simulation.MAX_REPLICAS,
                                               self.simulation.NUM_NODES))

        self.rl_placement_matrix = np.zeros((self.simulation.NUM_APPLICATIONS, self.simulation.MAX_IDS,
                                             self.simulation.NUM_SERVICES, self.simulation.MAX_REPLICAS,
                                             self.simulation.NUM_NODES))

        self.rl_user_delay = np.zeros(self.simulation.NUM_USER_REQUESTS)
        # fill with high values
        for u in range(self.simulation.NUM_USER_REQUESTS):
            self.rl_user_delay[u] = MAX_USER_DELAY

        self.rl_user_service_association = np.zeros(
            (self.simulation.NUM_USER_REQUESTS, self.simulation.NUM_APPLICATIONS,
             self.simulation.MAX_IDS, self.simulation.NUM_SERVICES,
             self.simulation.MAX_REPLICAS, self.simulation.NUM_NODES))

        reset_node_constraints(self)

        # Keywords for Reward calculation
        self.constraint_max_services_on_node = False
        self.constraint_add_service_max_replicas_reached = False
        self.constraint_terminate_service_without_deployed_services = False
        self.terminate_service = False
        self.constraint_node_capacity = False

        # Execute MILP
        # logging.info("Reset mode: MILP running...")
        # ilp.run(self.simulation)

        return np.array(self.get_state())

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    def take_action(self, action):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == MAX_STEPS:
            # logging.info('MAX STEPS achieved, ending ...')
            self.episode_over = True

        # Stop if current reward >= MAX REWARD
        if self.total_reward >= MAX_REWARD:
            # logging.info('MAX Reward achieved, ending ...')
            self.episode_over = True

        # ACTIONS IF/ELSE
        if action == ACTION_DO_NOTHING:
            # well do nothing!
            logging.info("SELECTED ACTION: DO NOTHING ...")

        # Service 1 Actions
        elif action == ACTION_ADD_SERVICE_1_TO_NODE_1:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 1 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 0, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_2:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 2 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 1, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_3:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 3 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 2, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_4:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 4 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 3, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_5:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 5 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 4, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_6:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 6 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 5, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_7:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 7 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 6, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_8:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 8 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 7, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_9:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 9 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 8, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_10:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 10 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 9, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_11:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 11 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 10, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_12:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 12 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 11, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_13:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 13 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 12, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_14:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 14 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 13, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_15:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 15 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 14, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_16:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 16 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 15, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_17:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 17 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 16, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_18:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 18 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 17, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_19:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 19 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 18, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_20:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 20 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 19, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_21:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 21 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 20, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_22:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 22 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 22, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_23:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 23 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 21, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_24:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 24 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 23, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_25:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 25 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 24, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_26:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 26 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 25, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_27:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 27 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 26, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_28:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 28 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 27, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_29:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 29 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 28, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_30:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 30 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 29, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_31:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 31 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 30, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_32:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 32 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 31, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_33:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 33 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 32, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_34:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 34 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 33, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_35:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 35 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 34, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_36:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 36 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 35, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_37:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 37 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 36, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_38:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 38 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 37, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_39:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 39 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 38, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_40:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 40 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 39, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_41:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 41 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 40, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_42:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 42 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 41, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_43:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 43 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 42, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_44:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 44 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 43, self)

        elif action == ACTION_ADD_SERVICE_1_TO_NODE_45:
            logging.info("SELECTED ACTION: ADD SERVICE 1 TO NODE 45 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 0, 44, self)

        # Service 2 Actions
        elif action == ACTION_ADD_SERVICE_2_TO_NODE_1:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 1 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 0, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_2:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 2 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 1, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_3:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 3 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 2, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_4:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 4 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 3, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_5:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 5 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 4, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_6:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 6 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 5, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_7:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 7 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 6, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_8:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 8 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 7, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_9:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 9 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 8, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_10:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 10 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 9, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_11:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 11 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 10, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_12:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 12 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 11, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_13:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 13 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 12, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_14:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 14 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 13, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_15:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 15 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 14, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_16:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 16 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 15, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_17:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 17 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 16, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_18:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 18 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 17, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_19:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 19 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 18, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_20:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 20 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 19, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_21:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 21 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 20, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_22:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 22 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 21, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_23:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 23 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 22, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_24:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 24 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 23, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_25:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 25 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 24, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_26:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 26 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 25, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_27:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 27 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 26, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_28:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 28 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 27, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_29:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 29 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 28, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_30:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 30 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 29, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_31:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 31 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 30, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_32:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 32 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 31, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_33:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 33 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 32, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_34:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 34 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 33, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_35:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 35 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 34, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_36:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 36 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 35, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_37:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 37 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 36, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_38:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 38 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 37, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_39:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 39 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 38, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_40:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 40 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 39, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_41:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 41 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 40, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_42:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 42 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 41, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_43:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 43 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 42, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_44:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 44 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 43, self)

        elif action == ACTION_ADD_SERVICE_2_TO_NODE_45:
            logging.info("SELECTED ACTION: ADD SERVICE 2 TO NODE 45 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 1, 44, self)

        # Service 3 Actions
        elif action == ACTION_ADD_SERVICE_3_TO_NODE_1:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 1 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 0, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_2:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 2 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 1, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_3:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 3 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 2, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_4:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 4 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 3, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_5:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 5 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 4, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_6:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 6 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 5, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_7:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 7 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 6, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_8:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 8 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 7, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_9:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 9 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 8, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_10:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 10 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 9, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_11:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 11 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 10, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_12:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 12 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 11, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_13:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 13 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 12, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_14:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 14 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 13, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_15:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 15 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 14, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_16:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 16 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 15, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_17:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 17 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 16, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_18:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 18 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 17, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_19:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 19 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 18, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_20:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 20 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 19, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_21:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 21 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 20, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_22:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 22 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 21, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_23:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 23 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 22, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_24:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 24 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 23, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_25:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 25 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 24, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_26:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 26 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 25, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_27:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 27 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 26, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_28:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 28 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 27, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_29:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 29 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 28, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_30:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 30 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 29, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_31:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 31 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 30, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_32:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 32 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 31, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_33:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 33 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 32, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_34:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 34 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 33, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_35:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 35 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 34, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_36:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 36 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 35, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_37:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 37 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 36, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_38:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 38 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 37, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_39:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 39 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 38, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_40:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 40 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 39, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_41:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 41 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 40, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_42:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 42 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 41, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_43:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 43 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 42, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_44:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 44 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 43, self)

        elif action == ACTION_ADD_SERVICE_3_TO_NODE_45:
            logging.info("SELECTED ACTION: ADD SERVICE 3 TO NODE 45 ...")

            # Deploy Service
            # a, id, s, n, self
            deploy_service(0, 0, 2, 44, self)

        # Terminate Service 1 Actions
        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_1:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 1 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 0, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_2:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 2 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 1, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_3:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 3 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 2, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_4:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 4 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 3, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_5:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 5 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 4, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_6:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 6 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 5, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_7:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 7 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 6, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_8:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 8 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 7, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_9:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 9 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 8, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_10:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 10 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 9, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_11:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 11 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 10, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_12:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 12 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 11, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_13:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 13 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 12, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_14:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 14 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 13, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_15:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 15 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 14, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_16:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 16 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 15, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_17:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 17 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 16, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_18:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 18 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 17, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_19:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 19 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 18, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_20:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 20 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 19, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_21:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 21 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 20, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_22:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 22 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 21, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_23:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 23 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 22, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_24:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 24 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 23, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_25:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 25 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 24, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_26:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 26 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 25, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_27:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 27 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 26, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_28:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 28 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 27, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_29:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 29 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 28, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_30:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 30 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 29, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_31:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 31 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 30, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_32:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 32 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 31, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_33:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 33 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 32, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_34:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 34 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 33, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_35:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 35 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 34, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_36:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 36 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 35, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_37:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 37 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 36, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_38:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 38 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 37, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_39:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 39 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 38, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_40:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 40 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 39, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_41:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 41 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 40, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_42:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 42 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 41, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_43:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 43 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 42, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_44:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 44 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 43, self)

        elif action == ACTION_TERMINATE_SERVICE_1_FROM_NODE_45:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 1 IN NODE 45 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 0, 44, self)

        # Terminate Service 2 Actions
        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_1:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 1 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 0, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_2:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 2 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 1, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_3:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 3 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 2, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_4:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 4 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 3, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_5:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 5 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 4, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_6:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 6 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 5, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_7:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 7 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 6, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_8:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 8 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 7, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_9:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 9 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 8, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_10:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 10 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 9, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_11:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 11 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 10, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_12:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 12 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 11, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_13:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 13 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 12, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_14:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 14 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 13, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_15:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 15 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 14, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_16:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 16 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 15, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_17:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 17 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 16, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_18:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 18 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 17, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_19:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 19 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 18, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_20:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 20 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 19, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_21:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 21 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 20, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_22:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 22 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 21, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_23:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 23 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 22, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_24:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 24 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 23, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_25:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 25 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 24, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_26:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 26 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 25, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_27:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 27 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 26, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_28:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 28 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 27, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_29:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 29 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 28, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_30:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 30 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 29, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_31:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 31 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 30, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_32:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 32 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 31, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_33:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 33 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 32, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_34:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 34 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 33, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_35:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 35 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 34, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_36:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 36 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 35, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_37:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 37 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 36, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_38:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 38 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 37, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_39:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 39 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 38, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_40:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 40 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 39, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_41:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 41 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 40, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_42:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 42 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 41, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_43:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 43 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 42, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_44:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 44 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 43, self)

        elif action == ACTION_TERMINATE_SERVICE_2_FROM_NODE_45:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 2 IN NODE 45 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 1, 44, self)

        # Terminate Service 3 Actions
        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_1:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 1 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 0, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_2:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 2 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 1, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_3:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 3 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 2, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_4:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 4 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 3, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_5:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 5 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 4, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_6:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 6 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 5, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_7:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 7 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 6, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_8:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 8 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 7, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_9:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 9 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 8, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_10:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 10 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 9, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_11:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 11 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 10, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_12:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 12 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 11, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_13:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 13 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 12, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_14:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 14 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 13, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_15:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 15 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 14, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_16:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 16 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 15, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_17:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 17 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 16, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_18:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 18 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 17, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_19:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 19 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 18, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_20:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 20 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 19, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_21:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 21 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 20, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_22:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 22 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 21, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_23:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 23 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 22, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_24:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 24 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 23, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_25:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 25 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 24, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_26:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 26 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 25, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_27:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 27 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 26, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_28:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 28 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 27, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_29:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 29 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 28, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_30:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 30 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 29, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_31:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 31 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 30, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_32:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 32 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 31, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_33:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 33 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 32, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_34:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 34 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 33, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_35:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 35 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 34, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_36:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 36 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 35, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_37:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 37 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 36, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_38:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 38 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 37, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_39:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 39 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 38, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_40:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 40 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 39, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_41:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 41 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 40, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_42:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 42 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 41, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_43:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 43 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 42, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_44:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 44 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 43, self)

        elif action == ACTION_TERMINATE_SERVICE_3_FROM_NODE_45:

            logging.info("SELECTED ACTION: TERMINATE SERVICE 3 IN NODE 45 ...")
            # Terminate Service
            # a, id, s, n, self
            terminate_service(0, 0, 2, 44, self)

        else:
            logging.info('\n Unrecognized Action: ' + str(action))

    def get_reward(self):
        """ Calculate Rewards """
        reward = 0

        number_users = self.number_users
        s1_rl = get_service_RL(0, self)
        s2_rl = get_service_RL(1, self)
        s3_rl = get_service_RL(2, self)
        s1_milp = get_service_MILP(0, self)
        s2_milp = get_service_MILP(1, self)
        s3_milp = get_service_MILP(2, self)
        cost_rl = get_min_cost_rl(self)
        cost_milp = get_min_cost_milp(self)
        acceptance = getPercentageRequests(s1_rl, s2_rl, s3_rl, s1_milp, s2_milp, s3_milp, number_users)

        s1_node_1 = get_number_deployed_service_instances_on_node(0, 0, self)
        s1_node_2 = get_number_deployed_service_instances_on_node(0, 1, self)
        s1_node_3 = get_number_deployed_service_instances_on_node(0, 2, self)
        s1_node_4 = get_number_deployed_service_instances_on_node(0, 3, self)
        s1_node_5 = get_number_deployed_service_instances_on_node(0, 4, self)
        s1_node_6 = get_number_deployed_service_instances_on_node(0, 5, self)
        s1_node_7 = get_number_deployed_service_instances_on_node(0, 6, self)
        s1_node_8 = get_number_deployed_service_instances_on_node(0, 7, self)
        s1_node_9 = get_number_deployed_service_instances_on_node(0, 8, self)
        s1_node_10 = get_number_deployed_service_instances_on_node(0, 9, self)
        s1_node_11 = get_number_deployed_service_instances_on_node(0, 10, self)
        s1_node_12 = get_number_deployed_service_instances_on_node(0, 11, self)
        s1_node_13 = get_number_deployed_service_instances_on_node(0, 12, self)
        s1_node_14 = get_number_deployed_service_instances_on_node(0, 13, self)
        s1_node_15 = get_number_deployed_service_instances_on_node(0, 14, self)
        s1_node_16 = get_number_deployed_service_instances_on_node(0, 15, self)
        s1_node_17 = get_number_deployed_service_instances_on_node(0, 16, self)
        s1_node_18 = get_number_deployed_service_instances_on_node(0, 17, self)
        s1_node_19 = get_number_deployed_service_instances_on_node(0, 18, self)
        s1_node_20 = get_number_deployed_service_instances_on_node(0, 19, self)
        s1_node_21 = get_number_deployed_service_instances_on_node(0, 20, self)
        s1_node_22 = get_number_deployed_service_instances_on_node(0, 21, self)
        s1_node_23 = get_number_deployed_service_instances_on_node(0, 22, self)
        s1_node_24 = get_number_deployed_service_instances_on_node(0, 23, self)
        s1_node_25 = get_number_deployed_service_instances_on_node(0, 24, self)
        s1_node_26 = get_number_deployed_service_instances_on_node(0, 25, self)
        s1_node_27 = get_number_deployed_service_instances_on_node(0, 26, self)
        s1_node_28 = get_number_deployed_service_instances_on_node(0, 27, self)
        s1_node_29 = get_number_deployed_service_instances_on_node(0, 28, self)
        s1_node_30 = get_number_deployed_service_instances_on_node(0, 29, self)
        s1_node_31 = get_number_deployed_service_instances_on_node(0, 30, self)
        s1_node_32 = get_number_deployed_service_instances_on_node(0, 31, self)
        s1_node_33 = get_number_deployed_service_instances_on_node(0, 32, self)
        s1_node_34 = get_number_deployed_service_instances_on_node(0, 33, self)
        s1_node_35 = get_number_deployed_service_instances_on_node(0, 34, self)
        s1_node_36 = get_number_deployed_service_instances_on_node(0, 35, self)
        s1_node_37 = get_number_deployed_service_instances_on_node(0, 36, self)
        s1_node_38 = get_number_deployed_service_instances_on_node(0, 37, self)
        s1_node_39 = get_number_deployed_service_instances_on_node(0, 38, self)
        s1_node_40 = get_number_deployed_service_instances_on_node(0, 39, self)
        s1_node_41 = get_number_deployed_service_instances_on_node(0, 40, self)
        s1_node_42 = get_number_deployed_service_instances_on_node(0, 41, self)
        s1_node_43 = get_number_deployed_service_instances_on_node(0, 42, self)
        s1_node_44 = get_number_deployed_service_instances_on_node(0, 43, self)
        s1_node_45 = get_number_deployed_service_instances_on_node(0, 44, self)

        s2_node_1 = get_number_deployed_service_instances_on_node(1, 0, self)
        s2_node_2 = get_number_deployed_service_instances_on_node(1, 1, self)
        s2_node_3 = get_number_deployed_service_instances_on_node(1, 2, self)
        s2_node_4 = get_number_deployed_service_instances_on_node(1, 3, self)
        s2_node_5 = get_number_deployed_service_instances_on_node(1, 4, self)
        s2_node_6 = get_number_deployed_service_instances_on_node(1, 5, self)
        s2_node_7 = get_number_deployed_service_instances_on_node(1, 6, self)
        s2_node_8 = get_number_deployed_service_instances_on_node(1, 7, self)
        s2_node_9 = get_number_deployed_service_instances_on_node(1, 8, self)
        s2_node_10 = get_number_deployed_service_instances_on_node(1, 9, self)
        s2_node_11 = get_number_deployed_service_instances_on_node(1, 10, self)
        s2_node_12 = get_number_deployed_service_instances_on_node(1, 11, self)
        s2_node_13 = get_number_deployed_service_instances_on_node(1, 12, self)
        s2_node_14 = get_number_deployed_service_instances_on_node(1, 13, self)
        s2_node_15 = get_number_deployed_service_instances_on_node(1, 14, self)
        s2_node_16 = get_number_deployed_service_instances_on_node(1, 15, self)
        s2_node_17 = get_number_deployed_service_instances_on_node(1, 16, self)
        s2_node_18 = get_number_deployed_service_instances_on_node(1, 17, self)
        s2_node_19 = get_number_deployed_service_instances_on_node(1, 18, self)
        s2_node_20 = get_number_deployed_service_instances_on_node(1, 19, self)
        s2_node_21 = get_number_deployed_service_instances_on_node(1, 20, self)
        s2_node_22 = get_number_deployed_service_instances_on_node(1, 21, self)
        s2_node_23 = get_number_deployed_service_instances_on_node(1, 22, self)
        s2_node_24 = get_number_deployed_service_instances_on_node(1, 23, self)
        s2_node_25 = get_number_deployed_service_instances_on_node(1, 24, self)
        s2_node_26 = get_number_deployed_service_instances_on_node(1, 25, self)
        s2_node_27 = get_number_deployed_service_instances_on_node(1, 26, self)
        s2_node_28 = get_number_deployed_service_instances_on_node(1, 27, self)
        s2_node_29 = get_number_deployed_service_instances_on_node(1, 28, self)
        s2_node_30 = get_number_deployed_service_instances_on_node(1, 29, self)
        s2_node_31 = get_number_deployed_service_instances_on_node(1, 30, self)
        s2_node_32 = get_number_deployed_service_instances_on_node(1, 31, self)
        s2_node_33 = get_number_deployed_service_instances_on_node(1, 32, self)
        s2_node_34 = get_number_deployed_service_instances_on_node(1, 33, self)
        s2_node_35 = get_number_deployed_service_instances_on_node(1, 34, self)
        s2_node_36 = get_number_deployed_service_instances_on_node(1, 35, self)
        s2_node_37 = get_number_deployed_service_instances_on_node(1, 36, self)
        s2_node_38 = get_number_deployed_service_instances_on_node(1, 37, self)
        s2_node_39 = get_number_deployed_service_instances_on_node(1, 38, self)
        s2_node_40 = get_number_deployed_service_instances_on_node(1, 39, self)
        s2_node_41 = get_number_deployed_service_instances_on_node(1, 40, self)
        s2_node_42 = get_number_deployed_service_instances_on_node(1, 41, self)
        s2_node_43 = get_number_deployed_service_instances_on_node(1, 42, self)
        s2_node_44 = get_number_deployed_service_instances_on_node(1, 43, self)
        s2_node_45 = get_number_deployed_service_instances_on_node(1, 44, self)

        s3_node_1 = get_number_deployed_service_instances_on_node(2, 0, self)
        s3_node_2 = get_number_deployed_service_instances_on_node(2, 1, self)
        s3_node_3 = get_number_deployed_service_instances_on_node(2, 2, self)
        s3_node_4 = get_number_deployed_service_instances_on_node(2, 3, self)
        s3_node_5 = get_number_deployed_service_instances_on_node(2, 4, self)
        s3_node_6 = get_number_deployed_service_instances_on_node(2, 5, self)
        s3_node_7 = get_number_deployed_service_instances_on_node(2, 6, self)
        s3_node_8 = get_number_deployed_service_instances_on_node(2, 7, self)
        s3_node_9 = get_number_deployed_service_instances_on_node(2, 8, self)
        s3_node_10 = get_number_deployed_service_instances_on_node(2, 9, self)
        s3_node_11 = get_number_deployed_service_instances_on_node(2, 10, self)
        s3_node_12 = get_number_deployed_service_instances_on_node(2, 11, self)
        s3_node_13 = get_number_deployed_service_instances_on_node(2, 12, self)
        s3_node_14 = get_number_deployed_service_instances_on_node(2, 13, self)
        s3_node_15 = get_number_deployed_service_instances_on_node(2, 14, self)
        s3_node_16 = get_number_deployed_service_instances_on_node(2, 15, self)
        s3_node_17 = get_number_deployed_service_instances_on_node(2, 16, self)
        s3_node_18 = get_number_deployed_service_instances_on_node(2, 17, self)
        s3_node_19 = get_number_deployed_service_instances_on_node(2, 18, self)
        s3_node_20 = get_number_deployed_service_instances_on_node(2, 19, self)
        s3_node_21 = get_number_deployed_service_instances_on_node(2, 20, self)
        s3_node_22 = get_number_deployed_service_instances_on_node(2, 21, self)
        s3_node_23 = get_number_deployed_service_instances_on_node(2, 22, self)
        s3_node_24 = get_number_deployed_service_instances_on_node(2, 23, self)
        s3_node_25 = get_number_deployed_service_instances_on_node(2, 24, self)
        s3_node_26 = get_number_deployed_service_instances_on_node(2, 25, self)
        s3_node_27 = get_number_deployed_service_instances_on_node(2, 26, self)
        s3_node_28 = get_number_deployed_service_instances_on_node(2, 27, self)
        s3_node_29 = get_number_deployed_service_instances_on_node(2, 28, self)
        s3_node_30 = get_number_deployed_service_instances_on_node(2, 29, self)
        s3_node_31 = get_number_deployed_service_instances_on_node(2, 30, self)
        s3_node_32 = get_number_deployed_service_instances_on_node(2, 31, self)
        s3_node_33 = get_number_deployed_service_instances_on_node(2, 32, self)
        s3_node_34 = get_number_deployed_service_instances_on_node(2, 33, self)
        s3_node_35 = get_number_deployed_service_instances_on_node(2, 34, self)
        s3_node_36 = get_number_deployed_service_instances_on_node(2, 35, self)
        s3_node_37 = get_number_deployed_service_instances_on_node(2, 36, self)
        s3_node_38 = get_number_deployed_service_instances_on_node(2, 37, self)
        s3_node_39 = get_number_deployed_service_instances_on_node(2, 38, self)
        s3_node_40 = get_number_deployed_service_instances_on_node(2, 39, self)
        s3_node_41 = get_number_deployed_service_instances_on_node(2, 40, self)
        s3_node_42 = get_number_deployed_service_instances_on_node(2, 41, self)
        s3_node_43 = get_number_deployed_service_instances_on_node(2, 42, self)
        s3_node_44 = get_number_deployed_service_instances_on_node(2, 43, self)
        s3_node_45 = get_number_deployed_service_instances_on_node(2, 44, self)

        n1_av_cpu = self.rl_node_av_cpu[0]
        n1_av_mem = self.rl_node_av_mem[0]
        n1_av_band = self.rl_node_av_band[0]
        n2_av_cpu = self.rl_node_av_cpu[1]
        n2_av_mem = self.rl_node_av_mem[1]
        n2_av_band = self.rl_node_av_band[1]
        n3_av_cpu = self.rl_node_av_cpu[2]
        n3_av_mem = self.rl_node_av_mem[2]
        n3_av_band = self.rl_node_av_band[2]
        n4_av_cpu = self.rl_node_av_cpu[3]
        n4_av_mem = self.rl_node_av_mem[3]
        n4_av_band = self.rl_node_av_band[3]
        n5_av_cpu = self.rl_node_av_cpu[4]
        n5_av_mem = self.rl_node_av_mem[4]
        n5_av_band = self.rl_node_av_band[4]
        n6_av_cpu = self.rl_node_av_cpu[5]
        n6_av_mem = self.rl_node_av_mem[5]
        n6_av_band = self.rl_node_av_band[5]
        n7_av_cpu = self.rl_node_av_cpu[6]
        n7_av_mem = self.rl_node_av_mem[6]
        n7_av_band = self.rl_node_av_band[6]
        n8_av_cpu = self.rl_node_av_cpu[7]
        n8_av_mem = self.rl_node_av_mem[7]
        n8_av_band = self.rl_node_av_band[7]
        n9_av_cpu = self.rl_node_av_cpu[8]
        n9_av_mem = self.rl_node_av_mem[8]
        n9_av_band = self.rl_node_av_band[8]
        n10_av_cpu = self.rl_node_av_cpu[9]
        n10_av_mem = self.rl_node_av_mem[9]
        n10_av_band = self.rl_node_av_band[9]
        n11_av_cpu = self.rl_node_av_cpu[10]
        n11_av_mem = self.rl_node_av_mem[10]
        n11_av_band = self.rl_node_av_band[10]
        n12_av_cpu = self.rl_node_av_cpu[11]
        n12_av_mem = self.rl_node_av_mem[11]
        n12_av_band = self.rl_node_av_band[11]
        n13_av_cpu = self.rl_node_av_cpu[12]
        n13_av_mem = self.rl_node_av_mem[12]
        n13_av_band = self.rl_node_av_band[12]
        n14_av_cpu = self.rl_node_av_cpu[13]
        n14_av_mem = self.rl_node_av_mem[13]
        n14_av_band = self.rl_node_av_band[13]
        n15_av_cpu = self.rl_node_av_cpu[14]
        n15_av_mem = self.rl_node_av_mem[14]
        n15_av_band = self.rl_node_av_band[14]
        n16_av_cpu = self.rl_node_av_cpu[15]
        n16_av_mem = self.rl_node_av_mem[15]
        n16_av_band = self.rl_node_av_band[15]
        n17_av_cpu = self.rl_node_av_cpu[16]
        n17_av_mem = self.rl_node_av_mem[16]
        n17_av_band = self.rl_node_av_band[16]
        n18_av_cpu = self.rl_node_av_cpu[17]
        n18_av_mem = self.rl_node_av_mem[17]
        n18_av_band = self.rl_node_av_band[17]
        n19_av_cpu = self.rl_node_av_cpu[18]
        n19_av_mem = self.rl_node_av_mem[18]
        n19_av_band = self.rl_node_av_band[18]
        n20_av_cpu = self.rl_node_av_cpu[19]
        n20_av_mem = self.rl_node_av_mem[19]
        n20_av_band = self.rl_node_av_band[19]
        n21_av_cpu = self.rl_node_av_cpu[20]
        n21_av_mem = self.rl_node_av_mem[20]
        n21_av_band = self.rl_node_av_band[20]
        n22_av_cpu = self.rl_node_av_cpu[21]
        n22_av_mem = self.rl_node_av_mem[21]
        n22_av_band = self.rl_node_av_band[21]
        n23_av_cpu = self.rl_node_av_cpu[22]
        n23_av_mem = self.rl_node_av_mem[22]
        n23_av_band = self.rl_node_av_band[22]
        n24_av_cpu = self.rl_node_av_cpu[23]
        n24_av_mem = self.rl_node_av_mem[23]
        n24_av_band = self.rl_node_av_band[23]
        n25_av_cpu = self.rl_node_av_cpu[24]
        n25_av_mem = self.rl_node_av_mem[24]
        n25_av_band = self.rl_node_av_band[24]
        n26_av_cpu = self.rl_node_av_cpu[25]
        n26_av_mem = self.rl_node_av_mem[25]
        n26_av_band = self.rl_node_av_band[25]
        n27_av_cpu = self.rl_node_av_cpu[26]
        n27_av_mem = self.rl_node_av_mem[26]
        n27_av_band = self.rl_node_av_band[26]
        n28_av_cpu = self.rl_node_av_cpu[27]
        n28_av_mem = self.rl_node_av_mem[27]
        n28_av_band = self.rl_node_av_band[27]
        n29_av_cpu = self.rl_node_av_cpu[28]
        n29_av_mem = self.rl_node_av_mem[28]
        n29_av_band = self.rl_node_av_band[28]
        n30_av_cpu = self.rl_node_av_cpu[29]
        n30_av_mem = self.rl_node_av_mem[29]
        n30_av_band = self.rl_node_av_band[29]
        n31_av_cpu = self.rl_node_av_cpu[30]
        n31_av_mem = self.rl_node_av_mem[30]
        n31_av_band = self.rl_node_av_band[30]
        n32_av_cpu = self.rl_node_av_cpu[31]
        n32_av_mem = self.rl_node_av_mem[31]
        n32_av_band = self.rl_node_av_band[31]
        n33_av_cpu = self.rl_node_av_cpu[32]
        n33_av_mem = self.rl_node_av_mem[32]
        n33_av_band = self.rl_node_av_band[32]
        n34_av_cpu = self.rl_node_av_cpu[33]
        n34_av_mem = self.rl_node_av_mem[33]
        n34_av_band = self.rl_node_av_band[33]
        n35_av_cpu = self.rl_node_av_cpu[34]
        n35_av_mem = self.rl_node_av_mem[34]
        n35_av_band = self.rl_node_av_band[34]
        n36_av_cpu = self.rl_node_av_cpu[35]
        n36_av_mem = self.rl_node_av_mem[35]
        n36_av_band = self.rl_node_av_band[35]
        n37_av_cpu = self.rl_node_av_cpu[36]
        n37_av_mem = self.rl_node_av_mem[36]
        n37_av_band = self.rl_node_av_band[36]
        n38_av_cpu = self.rl_node_av_cpu[37]
        n38_av_mem = self.rl_node_av_mem[37]
        n38_av_band = self.rl_node_av_band[37]
        n39_av_cpu = self.rl_node_av_cpu[38]
        n39_av_mem = self.rl_node_av_mem[38]
        n39_av_band = self.rl_node_av_band[38]
        n40_av_cpu = self.rl_node_av_cpu[39]
        n40_av_mem = self.rl_node_av_mem[39]
        n40_av_band = self.rl_node_av_band[39]
        n41_av_cpu = self.rl_node_av_cpu[40]
        n41_av_mem = self.rl_node_av_mem[40]
        n41_av_band = self.rl_node_av_band[40]
        n42_av_cpu = self.rl_node_av_cpu[41]
        n42_av_mem = self.rl_node_av_mem[41]
        n42_av_band = self.rl_node_av_band[41]
        n43_av_cpu = self.rl_node_av_cpu[42]
        n43_av_mem = self.rl_node_av_mem[42]
        n43_av_band = self.rl_node_av_band[42]
        n44_av_cpu = self.rl_node_av_cpu[43]
        n44_av_mem = self.rl_node_av_mem[43]
        n44_av_band = self.rl_node_av_band[43]
        n45_av_cpu = self.rl_node_av_cpu[44]
        n45_av_mem = self.rl_node_av_mem[44]
        n45_av_band = self.rl_node_av_band[44]

        logging.info(
            f"Obs: {int(self.current_step)} | "
            f"Requests: {int(number_users)} | "
            f"% Requests: {float(acceptance)} | "
            f"S1 RL: {int(s1_rl)} | "
            f"S2 RL: {int(s2_rl)} | "
            f"S3 RL: {int(s3_rl)} | "
            f"S1 MILP: {int(s1_milp)} | "
            f"S2 MILP: {int(s2_milp)} | "
            f"S3 MILP: {int(s3_milp)} | "
            f"Cost_RL: {float(cost_rl)} | "
            f"Cost MILP: {float(cost_milp)} | "
        )

        logging.info(
            f"S1 N1: {int(s1_node_1)} | "
            f"S1 N2: {int(s1_node_2)} | "
            f"S1 N3: {int(s1_node_3)} | "
            f"S1 N4: {int(s1_node_4)} | "
            f"S1 N5: {int(s1_node_5)} | "
            f"S1 N6: {int(s1_node_6)} | "
            f"S1 N7: {int(s1_node_7)} | "
            f"S1 N8: {int(s1_node_8)} | "
            f"S1 N9: {int(s1_node_9)} | "
            f"S1 N10: {int(s1_node_10)} | "
            f"S1 N11: {int(s1_node_11)} | "
            f"S1 N12: {int(s1_node_12)} | "
            f"S1 N13: {int(s1_node_13)} | "
            f"S1 N14: {int(s1_node_14)} | "
            f"S1 N15: {int(s1_node_15)} | "
        )

        logging.info(
            f"S1 N16: {int(s1_node_16)} | "
            f"S1 N17: {int(s1_node_17)} | "
            f"S1 N18: {int(s1_node_18)} | "
            f"S1 N19: {int(s1_node_19)} | "
            f"S1 N20: {int(s1_node_20)} | "
            f"S1 N21: {int(s1_node_21)} | "
            f"S1 N22: {int(s1_node_22)} | "
            f"S1 N23: {int(s1_node_23)} | "
            f"S1 N24: {int(s1_node_24)} | "
            f"S1 N25: {int(s1_node_25)} | "
            f"S1 N26: {int(s1_node_26)} | "
            f"S1 N27: {int(s1_node_27)} | "
            f"S1 N28: {int(s1_node_28)} | "
            f"S1 N29: {int(s1_node_29)} | "
            f"S1 N30: {int(s1_node_30)} | "
        )

        logging.info(
            f"S1 N31: {int(s1_node_31)} | "
            f"S1 N32: {int(s1_node_32)} | "
            f"S1 N33: {int(s1_node_33)} | "
            f"S1 N34: {int(s1_node_34)} | "
            f"S1 N35: {int(s1_node_35)} | "
            f"S1 N36: {int(s1_node_36)} | "
            f"S1 N37: {int(s1_node_37)} | "
            f"S1 N38: {int(s1_node_38)} | "
            f"S1 N39: {int(s1_node_39)} | "
            f"S1 N40: {int(s1_node_40)} | "
            f"S1 N41: {int(s1_node_41)} | "
            f"S1 N42: {int(s1_node_42)} | "
            f"S1 N43: {int(s1_node_43)} | "
            f"S1 N44: {int(s1_node_44)} | "
            f"S1 N45: {int(s1_node_45)} | "
        )

        logging.info(
            f"S2 N1: {int(s2_node_1)} | "
            f"S2 N2: {int(s2_node_2)} | "
            f"S2 N3: {int(s2_node_3)} | "
            f"S2 N4: {int(s2_node_4)} | "
            f"S2 N5: {int(s2_node_5)} | "
            f"S2 N6: {int(s2_node_6)} | "
            f"S2 N7: {int(s2_node_7)} | "
            f"S2 N8: {int(s2_node_8)} | "
            f"S2 N9: {int(s2_node_9)} | "
            f"S2 N10: {int(s2_node_10)} | "
            f"S2 N11: {int(s2_node_11)} | "
            f"S2 N12: {int(s2_node_12)} | "
            f"S2 N13: {int(s2_node_13)} | "
            f"S2 N14: {int(s2_node_14)} | "
            f"S2 N15: {int(s2_node_15)} | "
        )

        logging.info(
            f"S2 N16: {int(s2_node_16)} | "
            f"S2 N17: {int(s2_node_17)} | "
            f"S2 N18: {int(s2_node_18)} | "
            f"S2 N19: {int(s2_node_19)} | "
            f"S2 N20: {int(s2_node_20)} | "
            f"S2 N21: {int(s2_node_21)} | "
            f"S2 N22: {int(s2_node_22)} | "
            f"S2 N23: {int(s2_node_23)} | "
            f"S2 N24: {int(s2_node_24)} | "
            f"S2 N25: {int(s2_node_25)} | "
            f"S2 N26: {int(s2_node_26)} | "
            f"S2 N27: {int(s2_node_27)} | "
            f"S2 N28: {int(s2_node_28)} | "
            f"S2 N29: {int(s2_node_29)} | "
            f"S2 N30: {int(s2_node_30)} | "
        )

        logging.info(
            f"S2 N31: {int(s2_node_31)} | "
            f"S2 N32: {int(s2_node_32)} | "
            f"S2 N33: {int(s2_node_33)} | "
            f"S2 N34: {int(s2_node_34)} | "
            f"S2 N35: {int(s2_node_35)} | "
            f"S2 N36: {int(s2_node_36)} | "
            f"S2 N37: {int(s2_node_37)} | "
            f"S2 N38: {int(s2_node_38)} | "
            f"S2 N39: {int(s2_node_39)} | "
            f"S2 N40: {int(s2_node_40)} | "
            f"S2 N41: {int(s2_node_41)} | "
            f"S2 N42: {int(s2_node_42)} | "
            f"S2 N43: {int(s2_node_43)} | "
            f"S2 N44: {int(s2_node_44)} | "
            f"S2 N45: {int(s2_node_45)} | "
        )

        logging.info(
            f"S3 N1: {int(s3_node_1)} | "
            f"S3 N2: {int(s3_node_2)} | "
            f"S3 N3: {int(s3_node_3)} | "
            f"S3 N4: {int(s3_node_4)} | "
            f"S3 N5: {int(s3_node_5)} | "
            f"S3 N6: {int(s3_node_6)} | "
            f"S3 N7: {int(s3_node_7)} | "
            f"S3 N8: {int(s3_node_8)} | "
            f"S3 N9: {int(s3_node_9)} | "
            f"S3 N10: {int(s3_node_10)} | "
            f"S3 N11: {int(s3_node_11)} | "
            f"S3 N12: {int(s3_node_12)} | "
            f"S3 N13: {int(s3_node_13)} | "
            f"S3 N14: {int(s3_node_14)} | "
            f"S3 N15: {int(s3_node_15)} | "
        )

        logging.info(
            f"S3 N16: {int(s3_node_16)} | "
            f"S3 N17: {int(s3_node_17)} | "
            f"S3 N18: {int(s3_node_18)} | "
            f"S3 N19: {int(s3_node_19)} | "
            f"S3 N20: {int(s3_node_20)} | "
            f"S3 N21: {int(s3_node_21)} | "
            f"S3 N22: {int(s3_node_22)} | "
            f"S3 N23: {int(s3_node_23)} | "
            f"S3 N24: {int(s3_node_24)} | "
            f"S3 N25: {int(s3_node_25)} | "
            f"S3 N26: {int(s3_node_26)} | "
            f"S3 N27: {int(s3_node_27)} | "
            f"S3 N28: {int(s3_node_28)} | "
            f"S3 N29: {int(s3_node_29)} | "
            f"S3 N30: {int(s3_node_30)} | "
        )

        logging.info(
            f"S3 N31: {int(s3_node_31)} | "
            f"S3 N32: {int(s3_node_32)} | "
            f"S3 N33: {int(s3_node_33)} | "
            f"S3 N34: {int(s3_node_34)} | "
            f"S3 N35: {int(s3_node_35)} | "
            f"S3 N36: {int(s3_node_36)} | "
            f"S3 N37: {int(s3_node_37)} | "
            f"S3 N38: {int(s3_node_38)} | "
            f"S3 N39: {int(s3_node_39)} | "
            f"S3 N40: {int(s3_node_40)} | "
            f"S3 N41: {int(s3_node_41)} | "
            f"S3 N42: {int(s3_node_42)} | "
            f"S3 N43: {int(s3_node_43)} | "
            f"S3 N44: {int(s3_node_44)} | "
            f"S3 N45: {int(s3_node_45)} | "
        )

        logging.info(
            f"N1 CPU: {float(n1_av_cpu)} | "
            f"N2 CPU: {float(n2_av_cpu)} | "
            f"N3 CPU: {float(n3_av_cpu)} | "
            f"N4 CPU: {float(n4_av_cpu)} | "
            f"N5 CPU: {float(n5_av_cpu)} | "
            f"N6 CPU: {float(n6_av_cpu)} | "
            f"N7 CPU: {float(n7_av_cpu)} | "
            f"N8 CPU: {float(n8_av_cpu)} | "
            f"N9 CPU: {float(n9_av_cpu)} | "
            f"N10 CPU: {float(n10_av_cpu)} | "
            f"N11 CPU: {float(n11_av_cpu)} | "
            f"N12 CPU: {float(n12_av_cpu)} | "
            f"N13 CPU: {float(n13_av_cpu)} | "
            f"N14 CPU: {float(n14_av_cpu)} | "
            f"N15 CPU: {float(n15_av_cpu)} | "
        )

        logging.info(
            f"N16 CPU: {float(n16_av_cpu)} | "
            f"N17 CPU: {float(n17_av_cpu)} | "
            f"N18 CPU: {float(n18_av_cpu)} | "
            f"N19 CPU: {float(n19_av_cpu)} | "
            f"N20 CPU: {float(n20_av_cpu)} | "
            f"N21 CPU: {float(n21_av_cpu)} | "
            f"N22 CPU: {float(n22_av_cpu)} | "
            f"N23 CPU: {float(n23_av_cpu)} | "
            f"N24 CPU: {float(n24_av_cpu)} | "
            f"N25 CPU: {float(n25_av_cpu)} | "
            f"N26 CPU: {float(n26_av_cpu)} | "
            f"N27 CPU: {float(n27_av_cpu)} | "
            f"N28 CPU: {float(n28_av_cpu)} | "
            f"N29 CPU: {float(n29_av_cpu)} | "
            f"N30 CPU: {float(n30_av_cpu)} | "
        )

        logging.info(
            f"N31 CPU: {float(n31_av_cpu)} | "
            f"N32 CPU: {float(n32_av_cpu)} | "
            f"N33 CPU: {float(n33_av_cpu)} | "
            f"N34 CPU: {float(n34_av_cpu)} | "
            f"N35 CPU: {float(n35_av_cpu)} | "
            f"N36 CPU: {float(n36_av_cpu)} | "
            f"N37 CPU: {float(n37_av_cpu)} | "
            f"N38 CPU: {float(n38_av_cpu)} | "
            f"N39 CPU: {float(n39_av_cpu)} | "
            f"N40 CPU: {float(n40_av_cpu)} | "
            f"N41 CPU: {float(n41_av_cpu)} | "
            f"N42 CPU: {float(n42_av_cpu)} | "
            f"N43 CPU: {float(n43_av_cpu)} | "
            f"N44 CPU: {float(n44_av_cpu)} | "
            f"N45 CPU: {float(n45_av_cpu)} | "
        )

        logging.info(
            f"N1 MEM: {float(n1_av_mem)} | "
            f"N2 MEM: {float(n2_av_mem)} | "
            f"N3 MEM: {float(n3_av_mem)} | "
            f"N4 MEM: {float(n4_av_mem)} | "
            f"N5 MEM: {float(n5_av_mem)} | "
            f"N6 MEM: {float(n6_av_mem)} | "
            f"N7 MEM: {float(n7_av_mem)} | "
            f"N8 MEM: {float(n8_av_mem)} | "
            f"N9 MEM: {float(n9_av_mem)} | "
            f"N10 MEM: {float(n10_av_mem)} | "
            f"N11 MEM: {float(n11_av_mem)} | "
            f"N12 MEM: {float(n12_av_mem)} | "
            f"N13 MEM: {float(n13_av_mem)} | "
            f"N14 MEM: {float(n14_av_mem)} | "
            f"N15 MEM: {float(n15_av_mem)} | "
        )

        logging.info(
            f"N16 MEM: {float(n16_av_mem)} | "
            f"N17 MEM: {float(n17_av_mem)} | "
            f"N18 MEM: {float(n18_av_mem)} | "
            f"N19 MEM: {float(n19_av_mem)} | "
            f"N20 MEM: {float(n20_av_mem)} | "
            f"N21 MEM: {float(n21_av_mem)} | "
            f"N22 MEM: {float(n22_av_mem)} | "
            f"N23 MEM: {float(n23_av_mem)} | "
            f"N24 MEM: {float(n24_av_mem)} | "
            f"N25 MEM: {float(n25_av_mem)} | "
            f"N26 MEM: {float(n26_av_mem)} | "
            f"N27 MEM: {float(n27_av_mem)} | "
            f"N28 MEM: {float(n28_av_mem)} | "
            f"N29 MEM: {float(n29_av_mem)} | "
            f"N30 MEM: {float(n30_av_mem)} | "
        )

        logging.info(
            f"N31 MEM: {float(n31_av_mem)} | "
            f"N32 MEM: {float(n32_av_mem)} | "
            f"N33 MEM: {float(n33_av_mem)} | "
            f"N34 MEM: {float(n34_av_mem)} | "
            f"N35 MEM: {float(n35_av_mem)} | "
            f"N36 MEM: {float(n36_av_mem)} | "
            f"N37 MEM: {float(n37_av_mem)} | "
            f"N38 MEM: {float(n38_av_mem)} | "
            f"N39 MEM: {float(n39_av_mem)} | "
            f"N40 MEM: {float(n40_av_mem)} | "
            f"N41 MEM: {float(n41_av_mem)} | "
            f"N42 MEM: {float(n42_av_mem)} | "
            f"N43 MEM: {float(n43_av_mem)} | "
            f"N44 MEM: {float(n44_av_mem)} | "
            f"N45 MEM: {float(n45_av_mem)} | "
        )

        logging.info(
            f"N1 BAN: {float(n1_av_band)} | "
            f"N2 BAN: {float(n2_av_band)} | "
            f"N3 BAN: {float(n3_av_band)} | "
            f"N4 BAN: {float(n4_av_band)} | "
            f"N5 BAN: {float(n5_av_band)} | "
            f"N6 BAN: {float(n6_av_band)} | "
            f"N7 BAN: {float(n7_av_band)} | "
            f"N8 BAN: {float(n8_av_band)} | "
            f"N9 BAN: {float(n9_av_band)} | "
            f"N10 BAN: {float(n10_av_band)} | "
            f"N11 BAN: {float(n11_av_band)} | "
            f"N12 BAN: {float(n12_av_band)} | "
            f"N13 BAN: {float(n13_av_band)} | "
            f"N14 BAN: {float(n14_av_band)} | "
            f"N15 BAN: {float(n15_av_band)} | "
        )

        logging.info(
            f"N16 BAN: {float(n16_av_band)} | "
            f"N17 BAN: {float(n17_av_band)} | "
            f"N18 BAN: {float(n18_av_band)} | "
            f"N19 BAN: {float(n19_av_band)} | "
            f"N20 BAN: {float(n20_av_band)} | "
            f"N21 BAN: {float(n21_av_band)} | "
            f"N22 BAN: {float(n22_av_band)} | "
            f"N23 BAN: {float(n23_av_band)} | "
            f"N24 BAN: {float(n24_av_band)} | "
            f"N25 BAN: {float(n25_av_band)} | "
            f"N26 BAN: {float(n26_av_band)} | "
            f"N27 BAN: {float(n27_av_band)} | "
            f"N28 BAN: {float(n28_av_band)} | "
            f"N29 BAN: {float(n29_av_band)} | "
            f"N30 BAN: {float(n30_av_band)} | "
        )

        logging.info(
            f"N31 BAN: {float(n31_av_band)} | "
            f"N32 BAN: {float(n32_av_band)} | "
            f"N33 BAN: {float(n33_av_band)} | "
            f"N34 BAN: {float(n34_av_band)} | "
            f"N35 BAN: {float(n35_av_band)} | "
            f"N36 BAN: {float(n36_av_band)} | "
            f"N37 BAN: {float(n37_av_band)} | "
            f"N38 BAN: {float(n38_av_band)} | "
            f"N39 BAN: {float(n39_av_band)} | "
            f"N40 BAN: {float(n40_av_band)} | "
            f"N41 BAN: {float(n41_av_band)} | "
            f"N42 BAN: {float(n42_av_band)} | "
            f"N43 BAN: {float(n43_av_band)} | "
            f"N44 BAN: {float(n44_av_band)} | "
            f"N45 BAN: {float(n45_av_band)} | "
        )

        # Reward based on Keyword!
        if self.constraint_max_services_on_node:
            # logging.info("Reward[MAX Replicas]: Return 0!")
            return 0

        if self.constraint_terminate_service_without_deployed_services:
            # logging.info("Reward[Terminate Service]: Return 0!")
            return 0

        if self.constraint_add_service_max_replicas_reached:
            # logging.info("Reward[MAX Services]: Return 0!")
            return 0

        if self.constraint_node_capacity:
            # logging.info("Reward[MAX Node Capacity]: Return 0!")
            return 0

        # Service Reward Calculation
        reward += getServiceReward(s1_rl, s1_milp)
        reward += getServiceReward(s2_rl, s2_milp)
        reward += getServiceReward(s3_rl, s3_milp)

        # Return if acceptance = 0
        if acceptance == 0:
            return reward

        # Request Reward Calculation
        reward += int(acceptance)

        # Cost Reward Calculation
        reward += getCostReward(cost_rl, cost_milp)

        # Bonus reward
        if s1_rl == s1_milp and s2_rl == s2_milp and s3_rl == s3_milp:
            # MAX REWARD - Equal Energy efficiency
            if cost_rl == cost_milp:
                reward += 100

        return reward

    def get_state(self):
        # Observations: 100 Metrics!!
        # User Requests
        # % accepted requests
        # S1 RL
        # S2 RL
        # S3 RL
        # S1 MILP
        # S2 MILP
        # S3 MILP
        # Cost RL                      -> Compare with MILP solution
        # Cost MILP                    -> Compare with MILP solution
        # "Flag of S1 deployed in node 1" ... Flag of S3 deployed in node 15": 45 Observation Metrics
        # N1 CPU, N1 RAM, N1 Band ... N15 CPU, N15 RAM, N15 Band: 45 Observation Metrics

        ob = (self.number_users,
              getPercentageRequests(get_service_RL(0, self), get_service_RL(1, self),
                                    get_service_RL(2, self), get_service_MILP(0, self),
                                    get_service_MILP(1, self), get_service_MILP(2, self),
                                    self.number_users),
              get_service_RL(0, self), get_service_RL(1, self), get_service_RL(2, self),
              get_service_MILP(0, self), get_service_MILP(1, self), get_service_MILP(2, self),
              get_min_cost_rl(self), get_min_cost_milp(self),
              get_number_deployed_service_instances_on_node(0, 0, self),
              get_number_deployed_service_instances_on_node(0, 1, self),
              get_number_deployed_service_instances_on_node(0, 2, self),
              get_number_deployed_service_instances_on_node(0, 3, self),
              get_number_deployed_service_instances_on_node(0, 4, self),
              get_number_deployed_service_instances_on_node(0, 5, self),
              get_number_deployed_service_instances_on_node(0, 6, self),
              get_number_deployed_service_instances_on_node(0, 7, self),
              get_number_deployed_service_instances_on_node(0, 8, self),
              get_number_deployed_service_instances_on_node(0, 9, self),
              get_number_deployed_service_instances_on_node(0, 10, self),
              get_number_deployed_service_instances_on_node(0, 11, self),
              get_number_deployed_service_instances_on_node(0, 12, self),
              get_number_deployed_service_instances_on_node(0, 13, self),
              get_number_deployed_service_instances_on_node(0, 14, self),
              get_number_deployed_service_instances_on_node(0, 15, self),
              get_number_deployed_service_instances_on_node(0, 16, self),
              get_number_deployed_service_instances_on_node(0, 17, self),
              get_number_deployed_service_instances_on_node(0, 18, self),
              get_number_deployed_service_instances_on_node(0, 19, self),
              get_number_deployed_service_instances_on_node(0, 20, self),
              get_number_deployed_service_instances_on_node(0, 21, self),
              get_number_deployed_service_instances_on_node(0, 22, self),
              get_number_deployed_service_instances_on_node(0, 23, self),
              get_number_deployed_service_instances_on_node(0, 24, self),
              get_number_deployed_service_instances_on_node(0, 25, self),
              get_number_deployed_service_instances_on_node(0, 26, self),
              get_number_deployed_service_instances_on_node(0, 27, self),
              get_number_deployed_service_instances_on_node(0, 28, self),
              get_number_deployed_service_instances_on_node(0, 29, self),
              get_number_deployed_service_instances_on_node(0, 30, self),
              get_number_deployed_service_instances_on_node(0, 31, self),
              get_number_deployed_service_instances_on_node(0, 32, self),
              get_number_deployed_service_instances_on_node(0, 33, self),
              get_number_deployed_service_instances_on_node(0, 34, self),
              get_number_deployed_service_instances_on_node(0, 35, self),
              get_number_deployed_service_instances_on_node(0, 36, self),
              get_number_deployed_service_instances_on_node(0, 37, self),
              get_number_deployed_service_instances_on_node(0, 38, self),
              get_number_deployed_service_instances_on_node(0, 39, self),
              get_number_deployed_service_instances_on_node(0, 40, self),
              get_number_deployed_service_instances_on_node(0, 41, self),
              get_number_deployed_service_instances_on_node(0, 42, self),
              get_number_deployed_service_instances_on_node(0, 43, self),
              get_number_deployed_service_instances_on_node(0, 44, self),
              get_number_deployed_service_instances_on_node(1, 0, self),
              get_number_deployed_service_instances_on_node(1, 1, self),
              get_number_deployed_service_instances_on_node(1, 2, self),
              get_number_deployed_service_instances_on_node(1, 3, self),
              get_number_deployed_service_instances_on_node(1, 4, self),
              get_number_deployed_service_instances_on_node(1, 5, self),
              get_number_deployed_service_instances_on_node(1, 6, self),
              get_number_deployed_service_instances_on_node(1, 7, self),
              get_number_deployed_service_instances_on_node(1, 8, self),
              get_number_deployed_service_instances_on_node(1, 9, self),
              get_number_deployed_service_instances_on_node(1, 10, self),
              get_number_deployed_service_instances_on_node(1, 11, self),
              get_number_deployed_service_instances_on_node(1, 12, self),
              get_number_deployed_service_instances_on_node(1, 13, self),
              get_number_deployed_service_instances_on_node(1, 14, self),
              get_number_deployed_service_instances_on_node(1, 15, self),
              get_number_deployed_service_instances_on_node(1, 16, self),
              get_number_deployed_service_instances_on_node(1, 17, self),
              get_number_deployed_service_instances_on_node(1, 18, self),
              get_number_deployed_service_instances_on_node(1, 19, self),
              get_number_deployed_service_instances_on_node(1, 20, self),
              get_number_deployed_service_instances_on_node(1, 21, self),
              get_number_deployed_service_instances_on_node(1, 22, self),
              get_number_deployed_service_instances_on_node(1, 23, self),
              get_number_deployed_service_instances_on_node(1, 24, self),
              get_number_deployed_service_instances_on_node(1, 25, self),
              get_number_deployed_service_instances_on_node(1, 26, self),
              get_number_deployed_service_instances_on_node(1, 27, self),
              get_number_deployed_service_instances_on_node(1, 28, self),
              get_number_deployed_service_instances_on_node(1, 29, self),
              get_number_deployed_service_instances_on_node(1, 30, self),
              get_number_deployed_service_instances_on_node(1, 31, self),
              get_number_deployed_service_instances_on_node(1, 32, self),
              get_number_deployed_service_instances_on_node(1, 33, self),
              get_number_deployed_service_instances_on_node(1, 34, self),
              get_number_deployed_service_instances_on_node(1, 35, self),
              get_number_deployed_service_instances_on_node(1, 36, self),
              get_number_deployed_service_instances_on_node(1, 37, self),
              get_number_deployed_service_instances_on_node(1, 38, self),
              get_number_deployed_service_instances_on_node(1, 39, self),
              get_number_deployed_service_instances_on_node(1, 40, self),
              get_number_deployed_service_instances_on_node(1, 41, self),
              get_number_deployed_service_instances_on_node(1, 42, self),
              get_number_deployed_service_instances_on_node(1, 43, self),
              get_number_deployed_service_instances_on_node(1, 44, self),
              get_number_deployed_service_instances_on_node(2, 0, self),
              get_number_deployed_service_instances_on_node(2, 1, self),
              get_number_deployed_service_instances_on_node(2, 2, self),
              get_number_deployed_service_instances_on_node(2, 3, self),
              get_number_deployed_service_instances_on_node(2, 4, self),
              get_number_deployed_service_instances_on_node(2, 5, self),
              get_number_deployed_service_instances_on_node(2, 6, self),
              get_number_deployed_service_instances_on_node(2, 7, self),
              get_number_deployed_service_instances_on_node(2, 8, self),
              get_number_deployed_service_instances_on_node(2, 9, self),
              get_number_deployed_service_instances_on_node(2, 10, self),
              get_number_deployed_service_instances_on_node(2, 11, self),
              get_number_deployed_service_instances_on_node(2, 12, self),
              get_number_deployed_service_instances_on_node(2, 13, self),
              get_number_deployed_service_instances_on_node(2, 14, self),
              get_number_deployed_service_instances_on_node(2, 15, self),
              get_number_deployed_service_instances_on_node(2, 16, self),
              get_number_deployed_service_instances_on_node(2, 17, self),
              get_number_deployed_service_instances_on_node(2, 18, self),
              get_number_deployed_service_instances_on_node(2, 19, self),
              get_number_deployed_service_instances_on_node(2, 20, self),
              get_number_deployed_service_instances_on_node(2, 21, self),
              get_number_deployed_service_instances_on_node(2, 22, self),
              get_number_deployed_service_instances_on_node(2, 23, self),
              get_number_deployed_service_instances_on_node(2, 24, self),
              get_number_deployed_service_instances_on_node(2, 25, self),
              get_number_deployed_service_instances_on_node(2, 26, self),
              get_number_deployed_service_instances_on_node(2, 27, self),
              get_number_deployed_service_instances_on_node(2, 28, self),
              get_number_deployed_service_instances_on_node(2, 29, self),
              get_number_deployed_service_instances_on_node(2, 30, self),
              get_number_deployed_service_instances_on_node(2, 31, self),
              get_number_deployed_service_instances_on_node(2, 32, self),
              get_number_deployed_service_instances_on_node(2, 33, self),
              get_number_deployed_service_instances_on_node(2, 34, self),
              get_number_deployed_service_instances_on_node(2, 35, self),
              get_number_deployed_service_instances_on_node(2, 36, self),
              get_number_deployed_service_instances_on_node(2, 37, self),
              get_number_deployed_service_instances_on_node(2, 38, self),
              get_number_deployed_service_instances_on_node(2, 39, self),
              get_number_deployed_service_instances_on_node(2, 40, self),
              get_number_deployed_service_instances_on_node(2, 41, self),
              get_number_deployed_service_instances_on_node(2, 42, self),
              get_number_deployed_service_instances_on_node(2, 43, self),
              get_number_deployed_service_instances_on_node(2, 44, self),
              self.rl_node_av_cpu[0], self.rl_node_av_mem[0], self.rl_node_av_band[0],
              self.rl_node_av_cpu[1], self.rl_node_av_mem[1], self.rl_node_av_band[1],
              self.rl_node_av_cpu[2], self.rl_node_av_mem[2], self.rl_node_av_band[2],
              self.rl_node_av_cpu[3], self.rl_node_av_mem[3], self.rl_node_av_band[3],
              self.rl_node_av_cpu[4], self.rl_node_av_mem[4], self.rl_node_av_band[4],
              self.rl_node_av_cpu[5], self.rl_node_av_mem[5], self.rl_node_av_band[5],
              self.rl_node_av_cpu[6], self.rl_node_av_mem[6], self.rl_node_av_band[6],
              self.rl_node_av_cpu[7], self.rl_node_av_mem[7], self.rl_node_av_band[7],
              self.rl_node_av_cpu[8], self.rl_node_av_mem[8], self.rl_node_av_band[8],
              self.rl_node_av_cpu[9], self.rl_node_av_mem[9], self.rl_node_av_band[9],
              self.rl_node_av_cpu[10], self.rl_node_av_mem[10], self.rl_node_av_band[10],
              self.rl_node_av_cpu[11], self.rl_node_av_mem[11], self.rl_node_av_band[11],
              self.rl_node_av_cpu[12], self.rl_node_av_mem[12], self.rl_node_av_band[12],
              self.rl_node_av_cpu[13], self.rl_node_av_mem[13], self.rl_node_av_band[13],
              self.rl_node_av_cpu[14], self.rl_node_av_mem[14], self.rl_node_av_band[14],
              self.rl_node_av_cpu[15], self.rl_node_av_mem[15], self.rl_node_av_band[15],
              self.rl_node_av_cpu[16], self.rl_node_av_mem[16], self.rl_node_av_band[16],
              self.rl_node_av_cpu[17], self.rl_node_av_mem[17], self.rl_node_av_band[17],
              self.rl_node_av_cpu[18], self.rl_node_av_mem[18], self.rl_node_av_band[18],
              self.rl_node_av_cpu[19], self.rl_node_av_mem[19], self.rl_node_av_band[19],
              self.rl_node_av_cpu[20], self.rl_node_av_mem[20], self.rl_node_av_band[20],
              self.rl_node_av_cpu[21], self.rl_node_av_mem[21], self.rl_node_av_band[21],
              self.rl_node_av_cpu[22], self.rl_node_av_mem[22], self.rl_node_av_band[22],
              self.rl_node_av_cpu[23], self.rl_node_av_mem[23], self.rl_node_av_band[23],
              self.rl_node_av_cpu[24], self.rl_node_av_mem[24], self.rl_node_av_band[24],
              self.rl_node_av_cpu[25], self.rl_node_av_mem[25], self.rl_node_av_band[25],
              self.rl_node_av_cpu[26], self.rl_node_av_mem[26], self.rl_node_av_band[26],
              self.rl_node_av_cpu[27], self.rl_node_av_mem[27], self.rl_node_av_band[27],
              self.rl_node_av_cpu[28], self.rl_node_av_mem[28], self.rl_node_av_band[28],
              self.rl_node_av_cpu[29], self.rl_node_av_mem[29], self.rl_node_av_band[29],
              self.rl_node_av_cpu[30], self.rl_node_av_mem[30], self.rl_node_av_band[30],
              self.rl_node_av_cpu[31], self.rl_node_av_mem[31], self.rl_node_av_band[31],
              self.rl_node_av_cpu[32], self.rl_node_av_mem[32], self.rl_node_av_band[32],
              self.rl_node_av_cpu[33], self.rl_node_av_mem[33], self.rl_node_av_band[33],
              self.rl_node_av_cpu[34], self.rl_node_av_mem[34], self.rl_node_av_band[34],
              self.rl_node_av_cpu[35], self.rl_node_av_mem[35], self.rl_node_av_band[35],
              self.rl_node_av_cpu[36], self.rl_node_av_mem[36], self.rl_node_av_band[36],
              self.rl_node_av_cpu[37], self.rl_node_av_mem[37], self.rl_node_av_band[37],
              self.rl_node_av_cpu[38], self.rl_node_av_mem[38], self.rl_node_av_band[38],
              self.rl_node_av_cpu[39], self.rl_node_av_mem[39], self.rl_node_av_band[39],
              self.rl_node_av_cpu[40], self.rl_node_av_mem[40], self.rl_node_av_band[40],
              self.rl_node_av_cpu[41], self.rl_node_av_mem[41], self.rl_node_av_band[41],
              self.rl_node_av_cpu[42], self.rl_node_av_mem[42], self.rl_node_av_band[42],
              self.rl_node_av_cpu[43], self.rl_node_av_mem[43], self.rl_node_av_band[43],
              self.rl_node_av_cpu[44], self.rl_node_av_mem[44], self.rl_node_av_band[44],
              )

        return ob

    def user_request_management(self):

        if self.full_dynamic:
            choice = np.random.choice(np.arange(0, 3), p=[P_EQUAL, P_INCREASE, P_DECREASE])

            if choice == EQUAL:
                logging.info("EQUAL USER REQUESTS")

            elif choice == INCREASE:
                logging.info("INCREASE USER REQUESTS...")
                if self.number_users < self.simulation.MAX_USERS:
                    number_users = np.random.choice(np.arange(self.number_users, (self.simulation.MAX_USERS + 1)))

                    logging.info("New Number of User Requests: " + str(number_users))

                    self.simulation = self.list.search(number_users).get_sim()

                    previous_rl_user_service_association = self.rl_user_service_association
                    previous_rl_user_delay = self.rl_user_delay

                    # Create new Variables
                    self.rl_user_delay = np.zeros(self.simulation.NUM_USER_REQUESTS)

                    # fill with high values
                    for u in range(self.simulation.NUM_USER_REQUESTS):
                        self.rl_user_delay[u] = MAX_USER_DELAY

                    self.rl_user_service_association = np.zeros(
                        (self.simulation.NUM_USER_REQUESTS, self.simulation.NUM_APPLICATIONS,
                         self.simulation.MAX_IDS, self.simulation.NUM_SERVICES,
                         self.simulation.MAX_REPLICAS, self.simulation.NUM_NODES))

                    change_user_rl_variables(self, previous_rl_user_delay, previous_rl_user_service_association,
                                             self.number_users)

                    self.number_users = number_users
                else:
                    logging.info("Do Not change: MAX USER REQUESTS ACHIEVED")

            elif choice == DECREASE:
                logging.info("DECREASE USER REQUESTS...")
                if self.number_users > self.simulation.MIN_USERS:
                    number_users = np.random.choice(np.arange(self.simulation.MIN_USERS, self.number_users))

                    logging.info("New Number of User Requests: " + str(number_users))

                    self.simulation = self.list.search(number_users).get_sim()

                    previous_rl_user_service_association = self.rl_user_service_association
                    previous_rl_user_delay = self.rl_user_delay

                    # Create new Variables
                    self.rl_user_delay = np.zeros(self.simulation.NUM_USER_REQUESTS)

                    # fill with high values
                    for u in range(self.simulation.NUM_USER_REQUESTS):
                        self.rl_user_delay[u] = MAX_USER_DELAY

                    self.rl_user_service_association = np.zeros(
                        (self.simulation.NUM_USER_REQUESTS, self.simulation.NUM_APPLICATIONS,
                         self.simulation.MAX_IDS, self.simulation.NUM_SERVICES,
                         self.simulation.MAX_REPLICAS, self.simulation.NUM_NODES))

                    change_user_rl_variables(self, previous_rl_user_delay, previous_rl_user_service_association,
                                             number_users)

                    self.number_users = number_users
                else:
                    logging.info("Do Not change: MIN USER REQUESTS ACHIEVED")
            else:
                logging.info('\n Unrecognized Choice: ' + str(choice))
        else:
            if self.number_users < self.simulation.MAX_USERS:
                # Increase Requests by 1 no matter what
                number_users = self.number_users + 1
                logging.info("New Number of User Requests: " + str(number_users))

                self.simulation = self.list.search(number_users).get_sim()

                previous_rl_user_service_association = self.rl_user_service_association
                previous_rl_user_delay = self.rl_user_delay

                # Create new Variables
                self.rl_user_delay = np.zeros(self.simulation.NUM_USER_REQUESTS)

                # fill with high values
                for u in range(self.simulation.NUM_USER_REQUESTS):
                    self.rl_user_delay[u] = MAX_USER_DELAY

                self.rl_user_service_association = np.zeros(
                    (self.simulation.NUM_USER_REQUESTS, self.simulation.NUM_APPLICATIONS,
                     self.simulation.MAX_IDS, self.simulation.NUM_SERVICES,
                     self.simulation.MAX_REPLICAS, self.simulation.NUM_NODES))

                change_user_rl_variables(self, previous_rl_user_delay, previous_rl_user_service_association,
                                         self.number_users)

                self.number_users = number_users
            else:
                logging.info("Do Not change: MAX USER REQUESTS ACHIEVED")
        return
