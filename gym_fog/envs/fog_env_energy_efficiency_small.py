import csv
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_fog.envs.env_util import deploy_service, terminate_service, change_user_rl_variables, \
    get_min_cost_milp, get_min_cost_rl, get_ratio_service, get_number_deployed_service_instances_on_node, \
    get_number_deployed_service_instances, reset_node_constraints
from gym_fog.envs.linkedList import LinkedList, Node
from gym_fog.milp import ilp

from gym_fog.milp.simulation import Simulation

# General Variables defining the environment
from rl.util import plotting
from rl.util.save import save_ilp_csv

MAX_STEPS = 100
MAX_USER_DELAY = 10000
MAX_OBSERVATION_SPACE = 10000
MAX_REWARD = 10000

# User Discrete Event
EQUAL = 0
INCREASE = 1
DECREASE = 2
P_EQUAL = 0.35
P_INCREASE = 0.5
P_DECREASE = 0.15

# ACTIONS - Assuming 3 Services and 15 Nodes
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
ACTION_ADD_SERVICE_2_TO_NODE_1 = 16
ACTION_ADD_SERVICE_2_TO_NODE_2 = 17
ACTION_ADD_SERVICE_2_TO_NODE_3 = 18
ACTION_ADD_SERVICE_2_TO_NODE_4 = 19
ACTION_ADD_SERVICE_2_TO_NODE_5 = 20
ACTION_ADD_SERVICE_2_TO_NODE_6 = 21
ACTION_ADD_SERVICE_2_TO_NODE_7 = 22
ACTION_ADD_SERVICE_2_TO_NODE_8 = 23
ACTION_ADD_SERVICE_2_TO_NODE_9 = 24
ACTION_ADD_SERVICE_2_TO_NODE_10 = 25
ACTION_ADD_SERVICE_2_TO_NODE_11 = 26
ACTION_ADD_SERVICE_2_TO_NODE_12 = 27
ACTION_ADD_SERVICE_2_TO_NODE_13 = 28
ACTION_ADD_SERVICE_2_TO_NODE_14 = 29
ACTION_ADD_SERVICE_2_TO_NODE_15 = 30
ACTION_ADD_SERVICE_3_TO_NODE_1 = 31
ACTION_ADD_SERVICE_3_TO_NODE_2 = 32
ACTION_ADD_SERVICE_3_TO_NODE_3 = 33
ACTION_ADD_SERVICE_3_TO_NODE_4 = 34
ACTION_ADD_SERVICE_3_TO_NODE_5 = 35
ACTION_ADD_SERVICE_3_TO_NODE_6 = 36
ACTION_ADD_SERVICE_3_TO_NODE_7 = 37
ACTION_ADD_SERVICE_3_TO_NODE_8 = 38
ACTION_ADD_SERVICE_3_TO_NODE_9 = 39
ACTION_ADD_SERVICE_3_TO_NODE_10 = 40
ACTION_ADD_SERVICE_3_TO_NODE_11 = 41
ACTION_ADD_SERVICE_3_TO_NODE_12 = 42
ACTION_ADD_SERVICE_3_TO_NODE_13 = 43
ACTION_ADD_SERVICE_3_TO_NODE_14 = 44
ACTION_ADD_SERVICE_3_TO_NODE_15 = 45
ACTION_TERMINATE_SERVICE_1_FROM_NODE_1 = 46
ACTION_TERMINATE_SERVICE_1_FROM_NODE_2 = 47
ACTION_TERMINATE_SERVICE_1_FROM_NODE_3 = 48
ACTION_TERMINATE_SERVICE_1_FROM_NODE_4 = 49
ACTION_TERMINATE_SERVICE_1_FROM_NODE_5 = 50
ACTION_TERMINATE_SERVICE_1_FROM_NODE_6 = 51
ACTION_TERMINATE_SERVICE_1_FROM_NODE_7 = 52
ACTION_TERMINATE_SERVICE_1_FROM_NODE_8 = 53
ACTION_TERMINATE_SERVICE_1_FROM_NODE_9 = 54
ACTION_TERMINATE_SERVICE_1_FROM_NODE_10 = 55
ACTION_TERMINATE_SERVICE_1_FROM_NODE_11 = 56
ACTION_TERMINATE_SERVICE_1_FROM_NODE_12 = 57
ACTION_TERMINATE_SERVICE_1_FROM_NODE_13 = 58
ACTION_TERMINATE_SERVICE_1_FROM_NODE_14 = 59
ACTION_TERMINATE_SERVICE_1_FROM_NODE_15 = 60
ACTION_TERMINATE_SERVICE_2_FROM_NODE_1 = 61
ACTION_TERMINATE_SERVICE_2_FROM_NODE_2 = 62
ACTION_TERMINATE_SERVICE_2_FROM_NODE_3 = 63
ACTION_TERMINATE_SERVICE_2_FROM_NODE_4 = 64
ACTION_TERMINATE_SERVICE_2_FROM_NODE_5 = 65
ACTION_TERMINATE_SERVICE_2_FROM_NODE_6 = 66
ACTION_TERMINATE_SERVICE_2_FROM_NODE_7 = 67
ACTION_TERMINATE_SERVICE_2_FROM_NODE_8 = 68
ACTION_TERMINATE_SERVICE_2_FROM_NODE_9 = 69
ACTION_TERMINATE_SERVICE_2_FROM_NODE_10 = 70
ACTION_TERMINATE_SERVICE_2_FROM_NODE_11 = 71
ACTION_TERMINATE_SERVICE_2_FROM_NODE_12 = 72
ACTION_TERMINATE_SERVICE_2_FROM_NODE_13 = 73
ACTION_TERMINATE_SERVICE_2_FROM_NODE_14 = 74
ACTION_TERMINATE_SERVICE_2_FROM_NODE_15 = 75
ACTION_TERMINATE_SERVICE_3_FROM_NODE_1 = 76
ACTION_TERMINATE_SERVICE_3_FROM_NODE_2 = 77
ACTION_TERMINATE_SERVICE_3_FROM_NODE_3 = 78
ACTION_TERMINATE_SERVICE_3_FROM_NODE_4 = 79
ACTION_TERMINATE_SERVICE_3_FROM_NODE_5 = 80
ACTION_TERMINATE_SERVICE_3_FROM_NODE_6 = 81
ACTION_TERMINATE_SERVICE_3_FROM_NODE_7 = 82
ACTION_TERMINATE_SERVICE_3_FROM_NODE_8 = 83
ACTION_TERMINATE_SERVICE_3_FROM_NODE_9 = 84
ACTION_TERMINATE_SERVICE_3_FROM_NODE_10 = 85
ACTION_TERMINATE_SERVICE_3_FROM_NODE_11 = 86
ACTION_TERMINATE_SERVICE_3_FROM_NODE_12 = 87
ACTION_TERMINATE_SERVICE_3_FROM_NODE_13 = 88
ACTION_TERMINATE_SERVICE_3_FROM_NODE_14 = 89
ACTION_TERMINATE_SERVICE_3_FROM_NODE_15 = 90


def getRatioReward(ratio):
    if ratio == 0:
        return -5
    elif ratio == 1:
        return 5
    else:
        return -1


def getCostReward(cost_rl, cost_milp):
    # 4) MIN Cost
    # Lower Cost than the MILP -> Not possible
    if cost_rl < cost_milp:
        return -10
    # Up to 10% Higher Cost
    elif cost_milp <= cost_rl <= 1.10 * cost_milp:
        return 10
    elif 1.10 * cost_milp < cost_rl <= 1.25 * cost_milp:
        return -2
    elif 1.25 * cost_milp < cost_rl <= 1.75 * cost_milp:
        return -4
    elif 1.75 * cost_milp < cost_rl <= 2 * cost_milp:
        return -6
    elif 2 * cost_milp < cost_rl <= 3 * cost_milp:
        return -8
    elif 3 * cost_milp < cost_rl <= 4 * cost_milp:
        return -10
    else:  #>4
        return -20



class FogEnvEnergyEfficiencySmall(gym.Env):
    """A Fog Computing environment based on OpenAI gym"""

    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, name, number_users, dynamic=False):
        # Define action and observation space
        # They must be gym.spaces objects

        super(FogEnvEnergyEfficiencySmall, self).__init__()

        self.name = name
        self.dynamic = dynamic
        self.__version__ = "0.0.1"
        self.seed()

        logging.info("Fog Env Energy Efficiency Small - Version {}".format(self.__version__))

        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n
        self.num_actions = 91
        self.action_space = spaces.Discrete(self.num_actions)

        # Create MILP Simulation
        self.initial_users = number_users
        self.number_users = number_users
        self.initial_run = 1
        self.user_state = EQUAL
        self.simulation = Simulation("CPLEX_JOURNAL_JNCA", self.number_users, self.initial_run, "MIN_COST", "NONE",
                                     "NONE", "NONE", "NONE", "NONE")

        self.plot_ilp_time = np.array([0])

        # Observations: metrics - 54 Metrics!!
        # "number_users"
        # "ratio_service_1"              -> Compare with MILP solution
        # "ratio_service_2"              -> Compare with MILP solution
        # "ratio_service_3"              -> Compare with MILP solution
        # "cost_rl"                      -> Compare with MILP solution
        # "cost_milp"                    -> Compare with MILP solution

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([self.simulation.MAX_USERS,   # "number of Users"
                           MAX_OBSERVATION_SPACE,       # "ratio_service_1"
                           MAX_OBSERVATION_SPACE,       # "ratio_service_2"
                           MAX_OBSERVATION_SPACE,       # "ratio_service_3"
                           MAX_OBSERVATION_SPACE,       # "cost_rl"
                           MAX_OBSERVATION_SPACE,       # "cost_milp"

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
        # fill with high values
        for u in range(self.simulation.NUM_USER_REQUESTS):
            self.rl_user_delay[u] = MAX_USER_DELAY

        self.rl_user_service_association = np.zeros(
            (self.simulation.NUM_USER_REQUESTS, self.simulation.NUM_APPLICATIONS,
             self.simulation.MAX_IDS, self.simulation.NUM_SERVICES,
             self.simulation.MAX_REPLICAS, self.simulation.NUM_NODES))

        self.rl_node_av_cpu = np.zeros(self.simulation.NUM_NODES)
        self.rl_node_av_mem = np.zeros(self.simulation.NUM_NODES)
        self.rl_node_av_band = np.zeros(self.simulation.NUM_NODES)

        reset_node_constraints(self)

        # Info
        self.total_reward = None

        # episode over
        self.episode_over = False
        self.info = {}

        # Keywords for Reward calculation
        self.constraint_max_services_on_node = False
        self.constraint_add_service_max_replicas_reached = False
        self.constraint_terminate_service_without_deployed_services = False
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

        # Change User Requests at a 5 step interval
        if self.dynamic:
            if self.current_step % 5 == 0:
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

        else:
            logging.info('\n Unrecognized Action: ' + str(action))

    def get_reward(self):
        """ Calculate Rewards """
        reward = 0

        ratio_s1 = get_ratio_service(0, self)
        ratio_s2 = get_ratio_service(1, self)
        ratio_s3 = get_ratio_service(2, self)
        cost_rl = get_min_cost_rl(self)
        cost_milp = get_min_cost_milp(self)
        number_users = self.number_users

        logging.info(
            f"Obs: {int(self.current_step)} | "
            f"Requests: {int(number_users)} | "
            f"S1 Ratio: {float(ratio_s1)} | "
            f"S2 Ratio: {float(ratio_s2)} | "
            f"S3 Ratio: {float(ratio_s3)} | "
            f"Cost_RL: {float(cost_rl)} | "
            f"Cost MILP: {float(cost_milp)} | "
        )

        # Reward based on Keyword!
        if self.constraint_max_services_on_node:
            # logging.info("Reward[MAX Replicas]: Return -1!")
            return -1

        if self.constraint_terminate_service_without_deployed_services:
            # logging.info("Reward[Terminate Service]: Return -1!")
            return -1

        if self.constraint_add_service_max_replicas_reached:
            # logging.info("Reward[MAX Services]: Return -1!")
            return -1

        # Reward for Deploying Services
        #if self.deploy_service:
            #reward += 1

        # Reward for Terminating Services
        #if self.terminate_service:
            #reward -= 1

        # Cost Reward Calculation
        reward += getCostReward(cost_rl, cost_milp)

        if ratio_s1 == 1 and ratio_s2 == 1 and ratio_s3 == 1:
            # High REWARD
            if cost_rl > cost_milp:
                reward += 10
            # MAX REWARD - Equal Energy efficiency
            if cost_rl == cost_milp:
                reward += 100

        return reward

    def get_state(self):
        # Observations: metrics - 6 Metrics!!
        # "number_users"
        # "ratio_service_1"              -> Compare with MILP solution
        # "ratio_service_2"              -> Compare with MILP solution
        # "ratio_service_3"              -> Compare with MILP solution
        # "cost_rl"                      -> Compare with MILP solution
        # "cost_milp"                    -> Compare with MILP solution

        ob = (self.number_users, get_ratio_service(0, self), get_ratio_service(1, self), get_ratio_service(2, self),
              get_min_cost_rl(self), get_min_cost_milp(self))

        return ob

    def user_request_management(self):
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

        return
