import numpy as np
from gym_fog.milp.graph import Graph

from gym_fog.milp.util import fillApplicationIDs, fillBandwidthMatrix, fillDelayNodeLocationMatrix, \
    fillLatencyLocationMatrix, \
    fillNodeLatencyMatrix, fillInstanceMatrix, fillCommunicationMatrix, fillUsersApplicationMatrix, fillUserPositions, \
    fillUserLocationLatencyMatrix, fillDelayUserNodeMatrix, fillUserLocation, create_graph


class Simulation:
    # Objetive Variables
    op1 = ""
    op2 = ""
    op3 = ""
    op4 = ""
    op5 = ""
    op6 = ""
    obj1 = 0
    obj2 = 0
    obj3 = 0
    obj4 = 0
    obj5 = 0
    obj6 = 0

    # Duration Variables
    time_start = 0.0
    time_end = 0.0
    duration = 0.0
    duration_IT1 = 0.0
    duration_IT2 = 0.0
    duration_IT3 = 0.0
    duration_IT4 = 0.0
    duration_IT5 = 0.0
    duration_IT6 = 0.0
    total_execution = 0.0
    number_simulations = 0
    number_iterations = 0
    number_runs = 0
    run_number = 1

    # Variables MILP Model
    MAX_USERS = 50
    MIN_USERS = 1

    MAX_IDS = 1
    MAX_REPLICAS = 5

    NUM_APPLICATIONS = 1
    NUM_SERVICES = 3
    NUM_LOCATIONS = 9
    NUM_NODES = 45
    MIGRATION_FACTOR = 1.0
    NUM_USER_REQUESTS = 0
    NUM_SFC_SLOTS = 100

    node_names = ['worker1', 'worker2', 'worker3', 'worker4', 'worker5', 'worker6', 'worker7',
                  'worker8', 'worker9', 'worker10', 'worker11', 'worker12', 'worker13', 'worker14',
                  'worker15', 'worker16', 'worker17', 'worker18', 'worker19', 'worker20', 'worker21',
                  'worker22', 'worker23', 'worker24', 'worker25', 'worker26', 'worker27', 'worker28',
                  'worker29', 'worker30', 'worker31', 'worker32', 'worker33', 'worker34', 'worker35',
                  'worker36', 'worker37', 'worker38', 'worker39', 'worker40', 'worker41', 'worker42',
                  'worker43', 'worker44', 'worker45']

    location_names = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9']

    '''
    # Small simulation Variables
    node_cpu = [2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 6.0, 6.0, 8.0]
    node_mem = [4.0, 4.0, 2.0, 4.0, 2.0, 4.0, 4.0, 4.0, 2.0, 4.0, 4.0, 4.0, 16.0, 16.0, 24.0]
    node_band = [10.0, 10.0, 5.0, 10.0, 5.0, 10.0, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0, 30.0, 30.0, 30.0]
    weigh_node = [2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    node_location = [2, 2, 2, 0, 0, 0, 3, 3, 3, 1, 1, 1, 4, 4, 4]
    '''

    node_cpu = [1.0, 2.0, 2.0, 2.0, 1.0,  # L1
                1.0, 2.0, 2.0, 2.0, 1.0,  # L2
                1.0, 2.0, 2.0, 2.0, 1.0,  # L3
                1.0, 2.0, 2.0, 2.0, 1.0,  # L4
                8.0, 8.0, 8.0, 8.0, 8.0,  # L5
                1.0, 2.0, 2.0, 2.0, 1.0,  # L6
                1.0, 2.0, 2.0, 2.0, 1.0,  # L7
                1.0, 2.0, 2.0, 2.0, 1.0,  # L8
                1.0, 2.0, 2.0, 2.0, 1.0]  # L9

    node_mem = [2.0, 4.0, 4.0, 4.0, 2.0,  # L1
                2.0, 4.0, 4.0, 4.0, 2.0,  # L2
                2.0, 4.0, 4.0, 4.0, 2.0,  # L3
                2.0, 4.0, 4.0, 4.0, 2.0,  # L4
                16.0, 16.0, 16.0, 16.0, 16.0,  # L5
                2.0, 4.0, 4.0, 4.0, 2.0,   # L6
                2.0, 4.0, 4.0, 4.0, 2.0,   # L7
                2.0, 4.0, 4.0, 4.0, 2.0,   # L8
                2.0, 4.0, 4.0, 4.0, 2.0]   # L9

    node_band = [5.0, 10.0, 10.0, 10.0, 5.0,    # L1
                 5.0, 10.0, 10.0, 10.0, 5.0,    # L2
                 5.0, 10.0, 10.0, 10.0, 5.0,    # L3
                 5.0, 10.0, 10.0, 10.0, 5.0,    # L4
                 30.0, 30.0, 30.0, 30.0, 30.0,  # L5
                 5.0, 10.0, 10.0, 10.0, 5.0,    # L6
                 5.0, 10.0, 10.0, 10.0, 5.0,    # L7
                 5.0, 10.0, 10.0, 10.0, 5.0,    # L8
                 5.0, 10.0, 10.0, 10.0, 5.0]    # L9

    weigh_node = [1.0, 2.0, 2.0, 2.0, 1.0,  # L1
                1.0, 2.0, 2.0, 2.0, 1.0,  # L2
                1.0, 2.0, 2.0, 2.0, 1.0,  # L3
                1.0, 2.0, 2.0, 2.0, 1.0,  # L4
                3.0, 3.0, 3.0, 3.0, 3.0,  # L5
                1.0, 2.0, 2.0, 2.0, 1.0,  # L6
                1.0, 2.0, 2.0, 2.0, 1.0,  # L7
                1.0, 2.0, 2.0, 2.0, 1.0,  # L8
                1.0, 2.0, 2.0, 2.0, 1.0]  # L9

    node_location = [0, 0, 0, 0, 0, # L1
                    1, 1, 1, 1, 1,  # L2
                    2, 2, 2, 2, 2,  # L3
                    3, 3, 3, 3, 3,  # L4
                    4, 4, 4, 4, 4,  # L5
                    5, 5, 5, 5, 5,  # L6
                    6, 6, 6, 6, 6,  # L7
                    7, 7, 7, 7, 7,  # L8
                    8, 8, 8, 8, 8]  # L9

    application_ids = np.zeros(NUM_APPLICATIONS)

    service_names = ['Serv 1', 'Serv 2', 'Serv 3']

    #service_cpu = [0.25, 0.5, 0.5]
    #service_mem = [0.25, 1.0, 1.0]
    #service_band = [5.0, 8.0, 5.0]
    #service_sfc_first_pos = [1, 0, 0]
    #service_sfc_last_pos = [0, 0, 1]
    #service_slots = [5, 8, 5]

    service_cpu = [0.25, 0.5, 0.5]
    service_mem = [0.25, 1.0, 1.0]
    service_band = [4.0, 8.0, 5.0]

    service_sfc_first_pos = [1, 0, 0]
    service_sfc_last_pos = [0, 0, 1]
    service_slots = [4, 6, 4]

    number_replicas = 0

    def __init__(self, name, NUM_USER_REQUESTS, number_runs, op1, op2, op3, op4, op5, op6):
        self.name = name
        self.NUM_USER_REQUESTS = NUM_USER_REQUESTS
        self.number_runs = number_runs
        self.op1 = op1
        self.op2 = op2
        self.op3 = op3
        self.op4 = op4
        self.op5 = op5
        self.op6 = op6
        self.create_simulation()

    def create_simulation(self):
        self.graph = Graph()
        create_graph(self.graph)
        self.user_location = []
        self.user_cost = np.zeros((self.NUM_USER_REQUESTS, self.NUM_APPLICATIONS))
        self.user_application = np.zeros((self.NUM_USER_REQUESTS, self.NUM_APPLICATIONS))
        self.user_location_latency = np.zeros((self.NUM_USER_REQUESTS, self.NUM_LOCATIONS))
        self.delay_user_node = np.zeros((self.NUM_USER_REQUESTS, self.NUM_NODES))

        self.xUser = np.zeros(self.NUM_USER_REQUESTS)
        self.yUser = np.zeros(self.NUM_USER_REQUESTS)

        self.band_matrix = np.zeros((self.NUM_NODES, self.NUM_NODES))
        self.delay_node_location = np.zeros((self.NUM_NODES, self.NUM_LOCATIONS))
        self.location_latency = np.zeros((self.NUM_LOCATIONS, self.NUM_LOCATIONS))
        self.node_latency = np.zeros((self.NUM_NODES, self.NUM_NODES))

        self.instance = np.zeros((self.NUM_APPLICATIONS, self.NUM_SERVICES))
        self.sfc_instance = np.zeros((self.NUM_SERVICES, self.NUM_SERVICES))
        self.communication = np.zeros((self.NUM_SERVICES, self.NUM_SERVICES))

        self.previous_placement_matrix = np.zeros(
            (self.NUM_APPLICATIONS, self.MAX_IDS, self.NUM_SERVICES, self.MAX_REPLICAS, self.NUM_NODES))

        self.milp_solution_previous_placement_matrix = np.zeros(
            (self.NUM_APPLICATIONS, self.MAX_IDS, self.NUM_SERVICES, self.MAX_REPLICAS, self.NUM_NODES))
        self.milp_solution_placement_matrix = np.zeros(
            (self.NUM_APPLICATIONS, self.MAX_IDS, self.NUM_SERVICES, self.MAX_REPLICAS, self.NUM_NODES))

        self.milp_solution_user_service_association = np.zeros((self.NUM_USER_REQUESTS, self.NUM_APPLICATIONS,
                                                                self.MAX_IDS, self.NUM_SERVICES, self.MAX_REPLICAS,
                                                                self.NUM_NODES))
        self.milp_solution_user_delay = np.zeros(self.NUM_USER_REQUESTS)

        # self.rl_solution_placement_matrix = np.zeros((self.NUM_APPLICATIONS, self.MAX_IDS, self.NUM_SERVICES, self.MAX_REPLICAS, self.NUM_NODES))
        # self.rl_solution_user_service_association = np.zeros((self.NUM_USER_REQUESTS, self.NUM_APPLICATIONS, self.MAX_IDS, self.NUM_SERVICES, self.MAX_REPLICAS, self.NUM_NODES))

        fillUserLocation(self.user_location, self.NUM_USER_REQUESTS)
        fillApplicationIDs(self.application_ids, self.NUM_APPLICATIONS, self.MAX_IDS)
        fillBandwidthMatrix(self.band_matrix, self.NUM_NODES, self.node_band)
        fillDelayNodeLocationMatrix(self.delay_node_location, self.graph, self.NUM_NODES, self.NUM_LOCATIONS,
                                    self.node_names, self.location_names)
        fillLatencyLocationMatrix(self.location_latency, self.graph, self.NUM_LOCATIONS, self.location_names)
        fillNodeLatencyMatrix(self.node_latency, self.NUM_NODES, self.node_location, self.delay_node_location)
        fillInstanceMatrix(self.instance, self.sfc_instance)
        fillCommunicationMatrix(self.communication, self.NUM_SERVICES, self.sfc_instance, self.service_band)
        fillUsersApplicationMatrix(self.user_application, self.user_cost, self.NUM_APPLICATIONS, self.NUM_USER_REQUESTS)
        fillUserPositions(self.user_location, self.xUser, self.yUser, self.NUM_USER_REQUESTS)
        fillUserLocationLatencyMatrix(self.user_location_latency, self.location_latency, self.user_location,
                                      self.NUM_USER_REQUESTS,
                                      self.NUM_LOCATIONS)
        fillDelayUserNodeMatrix(self.user_location_latency, self.delay_node_location, self.delay_user_node,
                                self.node_location,
                                self.NUM_NODES, self.NUM_USER_REQUESTS)

    def increase_number_users(self, new_request):
        self.NUM_USER_REQUESTS = new_request

        self.user_location = []
        self.user_cost = np.zeros((self.NUM_USER_REQUESTS, self.NUM_APPLICATIONS))
        self.user_application = np.zeros((self.NUM_USER_REQUESTS, self.NUM_APPLICATIONS))
        self.user_location_latency = np.zeros((self.NUM_USER_REQUESTS, self.NUM_LOCATIONS))
        self.delay_user_node = np.zeros((self.NUM_USER_REQUESTS, self.NUM_NODES))

        self.xUser = np.zeros(self.NUM_USER_REQUESTS)
        self.yUser = np.zeros(self.NUM_USER_REQUESTS)

        self.milp_solution_user_service_association = np.zeros((self.NUM_USER_REQUESTS, self.NUM_APPLICATIONS,
                                                                self.MAX_IDS, self.NUM_SERVICES, self.MAX_REPLICAS,
                                                                self.NUM_NODES))
        self.milp_solution_user_delay = np.zeros(self.NUM_USER_REQUESTS)

        self.rl_solution_user_service_association = np.zeros((self.NUM_USER_REQUESTS, self.NUM_APPLICATIONS,
                                                              self.MAX_IDS, self.NUM_SERVICES, self.MAX_REPLICAS,
                                                              self.NUM_NODES))

        fillUserLocation(self.user_location, self.NUM_USER_REQUESTS)
        fillUsersApplicationMatrix(self.user_application, self.user_cost, self.NUM_APPLICATIONS, self.NUM_USER_REQUESTS)
        fillUserPositions(self.user_location, self.xUser, self.yUser, self.NUM_USER_REQUESTS)
        fillUserLocationLatencyMatrix(self.user_location_latency, self.location_latency, self.user_location,
                                      self.NUM_USER_REQUESTS,
                                      self.NUM_LOCATIONS)
        fillDelayUserNodeMatrix(self.user_location_latency, self.delay_node_location, self.delay_user_node,
                                self.node_location,
                                self.NUM_NODES, self.NUM_USER_REQUESTS)

    def change_user_locations(self):
        self.user_location = []
        fillUserLocation(self.user_location, self.NUM_USER_REQUESTS)
        fillUsersApplicationMatrix(self.user_application, self.user_cost, self.NUM_APPLICATIONS, self.NUM_USER_REQUESTS)
        fillUserPositions(self.user_location, self.xUser, self.yUser, self.NUM_USER_REQUESTS)
        fillUserLocationLatencyMatrix(self.user_location_latency, self.location_latency, self.user_location,
                                      self.NUM_USER_REQUESTS,
                                      self.NUM_LOCATIONS)

        fillDelayUserNodeMatrix(self.user_location_latency, self.delay_node_location, self.delay_user_node,
                                self.node_location,
                                self.NUM_NODES, self.NUM_USER_REQUESTS)


