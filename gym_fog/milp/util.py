from random import randrange
from gym_fog.milp.graph import dijsktra


def fillUserLocation(user_location, NUM_USER_REQUESTS):
    for i in range(NUM_USER_REQUESTS):
        # user_location.append(randrange(5))  # Integer from 0 to 4 inclusive -> 5 Locations
        user_location.append(randrange(9))  # Integer from 0 to 8 inclusive -> 9 Locations


def fillApplicationIDs(application_ids, NUM_APPLICATIONS, MAX_IDS):
    # Populate Bandwidth Matrix
    for a in range(NUM_APPLICATIONS):
        application_ids[a] = MAX_IDS


def fillBandwidthMatrix(bandwidthMatrix, NUM_NODES, node_band):
    # Populate Bandwidth Matrix
    for n1 in range(NUM_NODES):
        for n2 in range(NUM_NODES):
            if n1 != n2:
                bandwidthMatrix[n1][n2] = min(node_band[n1], node_band[n2])
            else:
                bandwidthMatrix[n1][n2] = node_band[n1]


def create_graph(graph):
    # 45 Nodes
    graph.add_edge("worker1", "l1", 5)
    graph.add_edge("worker2", "l1", 3)
    graph.add_edge("worker3", "l1", 3)
    graph.add_edge("worker4", "l1", 3)
    graph.add_edge("worker5", "l1", 5)

    graph.add_edge("worker6", "l2", 5)
    graph.add_edge("worker7", "l2", 3)
    graph.add_edge("worker8", "l2", 3)
    graph.add_edge("worker9", "l2", 3)
    graph.add_edge("worker10", "l2", 5)

    graph.add_edge("worker11", "l3", 5)
    graph.add_edge("worker12", "l3", 3)
    graph.add_edge("worker13", "l3", 3)
    graph.add_edge("worker14", "l3", 3)
    graph.add_edge("worker15", "l3", 5)

    graph.add_edge("worker16", "l4", 5)
    graph.add_edge("worker17", "l4", 3)
    graph.add_edge("worker18", "l4", 3)
    graph.add_edge("worker19", "l4", 3)
    graph.add_edge("worker20", "l4", 5)

    graph.add_edge("worker21", "l5", 1)
    graph.add_edge("worker22", "l5", 1)
    graph.add_edge("worker23", "l5", 1)
    graph.add_edge("worker24", "l5", 1)
    graph.add_edge("worker25", "l5", 1)

    graph.add_edge("worker26", "l6", 5)
    graph.add_edge("worker27", "l6", 3)
    graph.add_edge("worker28", "l6", 3)
    graph.add_edge("worker29", "l6", 3)
    graph.add_edge("worker30", "l6", 5)

    graph.add_edge("worker31", "l7", 5)
    graph.add_edge("worker32", "l7", 3)
    graph.add_edge("worker33", "l7", 3)
    graph.add_edge("worker34", "l7", 3)
    graph.add_edge("worker35", "l7", 5)

    graph.add_edge("worker36", "l8", 5)
    graph.add_edge("worker37", "l8", 3)
    graph.add_edge("worker38", "l8", 3)
    graph.add_edge("worker39", "l8", 3)
    graph.add_edge("worker40", "l8", 5)

    graph.add_edge("worker41", "l9", 5)
    graph.add_edge("worker42", "l9", 3)
    graph.add_edge("worker43", "l9", 3)
    graph.add_edge("worker44", "l9", 3)
    graph.add_edge("worker45", "l9", 5)

    graph.add_edge("l1", "sw1", 1)
    graph.add_edge("l2", "sw1", 1)

    graph.add_edge("l3", "sw2", 1)
    graph.add_edge("l6", "sw2", 1)

    graph.add_edge("l4", "sw3", 1)
    graph.add_edge("l7", "sw3", 1)

    graph.add_edge("l8", "sw4", 1)
    graph.add_edge("l9", "sw4", 1)

    graph.add_edge("sw1", "l5", 25)
    graph.add_edge("sw4", "l5", 25)

    graph.add_edge("sw1", "sw2", 15)
    graph.add_edge("sw1", "sw3", 25)
    graph.add_edge("sw3", "sw4", 15)
    graph.add_edge("sw4", "sw2", 25)


def fillDelayNodeLocationMatrix(delay_node_location, graph, NUM_NODES, NUM_LOCATIONS, node_names, location_names):
    # Journal JNCA 15 Nodes
    '''
    delay_node_location[0][0] = 18.0  # work1 Location A
    delay_node_location[0][1] = 33.0  # work1 Location B
    delay_node_location[0][2] = 3.0  # work1 Location C
    delay_node_location[0][3] = 18.0  # work1 Location D
    delay_node_location[0][4] = 43.0  # work1 Location E
    delay_node_location[1][0] = 18.0  # work2 Location A
    delay_node_location[1][1] = 33.0  # work2 Location B
    delay_node_location[1][2] = 3.0  # work2 Location C
    delay_node_location[1][3] = 18.0  # work2 Location D
    delay_node_location[1][4] = 43.0  # work2 Location E
    delay_node_location[2][0] = 20.0  # work3 Location A
    delay_node_location[2][1] = 35.0  # work3 Location B
    delay_node_location[2][2] = 5.0  # work3 Location C
    delay_node_location[2][3] = 20.0  # work3 Location D
    delay_node_location[2][4] = 45.0  # work3 Location E

    delay_node_location[3][0] = 3.0  # work4 Location A
    delay_node_location[3][1] = 18.0  # work4 Location B
    delay_node_location[3][2] = 18.0  # work4 Location C
    delay_node_location[3][3] = 33.0  # work4 Location D
    delay_node_location[3][4] = 28.0  # work4 Location E
    delay_node_location[4][0] = 5.0  # work5 Location A
    delay_node_location[4][1] = 20.0  # work5 Location B
    delay_node_location[4][2] = 20.0  # work5 Location C
    delay_node_location[4][3] = 35.0  # work5 Location D
    delay_node_location[4][4] = 30.0  # work5 Location E
    delay_node_location[5][0] = 3.0  # work6 Location A
    delay_node_location[5][1] = 18.0  # work6 Location B
    delay_node_location[5][2] = 18.0  # work6 Location C
    delay_node_location[5][3] = 33.0  # work6 Location D
    delay_node_location[5][4] = 28.0  # work6 Location E

    delay_node_location[6][0] = 33.0  # work7 Location A
    delay_node_location[6][1] = 18.0  # work7 Location B
    delay_node_location[6][2] = 18.0  # work7 Location C
    delay_node_location[6][3] = 3.0  # work7 Location D
    delay_node_location[6][4] = 28.0  # work7 Location E
    delay_node_location[7][0] = 33.0  # work8 Location A
    delay_node_location[7][1] = 18.0  # work8 Location B
    delay_node_location[7][2] = 18.0  # work8 Location C
    delay_node_location[7][3] = 3.0  # work8 Location D
    delay_node_location[7][4] = 28.0  # work8 Location E
    delay_node_location[8][0] = 35.0  # work9 Location A
    delay_node_location[8][1] = 20.0  # work9 Location B
    delay_node_location[8][2] = 20.0  # work9 Location C
    delay_node_location[8][3] = 5.0  # work9 Location D
    delay_node_location[8][4] = 30.0  # work9 Location E

    delay_node_location[9][0] = 18.0  # work10 Location A
    delay_node_location[9][1] = 3.0  # work10 Location B
    delay_node_location[9][2] = 33.0  # work10 Location C
    delay_node_location[9][3] = 18.0  # work10 Location D
    delay_node_location[9][4] = 43.0  # work10 Location E
    delay_node_location[10][0] = 18.0  # work11 Location A
    delay_node_location[10][1] = 3.0  # work11 Location B
    delay_node_location[10][2] = 33.0  # work11 Location C
    delay_node_location[10][3] = 18.0  # work11 Location D
    delay_node_location[10][4] = 43.0  # work11 Location E
    delay_node_location[11][0] = 18.0  # work12 Location A
    delay_node_location[11][1] = 3.0  # work12 Location B
    delay_node_location[11][2] = 33.0  # work12 Location C
    delay_node_location[11][3] = 18.0  # work12 Location D
    delay_node_location[11][4] = 43.0  # work12 Location E

    delay_node_location[12][0] = 26.0  # work13 Location A
    delay_node_location[12][1] = 41.0  # work13 Location B
    delay_node_location[12][2] = 41.0  # work13 Location C
    delay_node_location[12][3] = 26.0  # work13 Location D
    delay_node_location[12][4] = 1.0  # work13 Location E
    delay_node_location[13][0] = 26.0  # work14 Location A
    delay_node_location[13][1] = 41.0  # work14 Location B
    delay_node_location[13][2] = 41.0  # work14 Location C
    delay_node_location[13][3] = 26.0  # work14 Location D
    delay_node_location[13][4] = 1.0  # work14 Location E
    delay_node_location[14][0] = 26.0  # Master Location A
    delay_node_location[14][1] = 41.0  # Master Location B
    delay_node_location[14][2] = 41.0  # Master Location C
    delay_node_location[14][3] = 26.0  # Master Location D
    delay_node_location[14][4] = 1.0  # Master Location E
    '''

    # DDQN Test 45 Nodes - 9 Locations
    for n in range(NUM_NODES):
        for l in range(NUM_LOCATIONS):
            #print(node_names[n])
            #print(location_names[l])
            path, weight = dijsktra(graph, node_names[n], location_names[l])
            delay_node_location[n][l] = weight
            #print(weight)


def fillLatencyLocationMatrix(location_latency, graph, NUM_LOCATIONS, location_names):
    # Journal ILP
    '''
    location_latency[0][0] = 0.0  # A - A
    location_latency[0][1] = 15.0  # A - B
    location_latency[0][2] = 15.0  # A - C
    location_latency[0][3] = 30.0  # A - D
    location_latency[0][4] = 25.0  # A - E

    location_latency[1][0] = 15.0  # B - A
    location_latency[1][1] = 0.0  # B - B
    location_latency[1][2] = 30.0  # B - C
    location_latency[1][3] = 15.0  # B - D
    location_latency[1][4] = 40.0  # B - E

    location_latency[2][0] = 15.0  # C - A
    location_latency[2][1] = 30.0  # C - B
    location_latency[2][2] = 0.0  # C - C
    location_latency[2][3] = 15.0  # C - D
    location_latency[2][4] = 40.0  # C - E

    location_latency[3][0] = 30.0  # D - A
    location_latency[3][1] = 15.0  # D - B
    location_latency[3][2] = 15.0  # D - C
    location_latency[3][3] = 0.0  # D - D
    location_latency[3][4] = 25.0  # D - E

    location_latency[4][0] = 25.0  # E - A
    location_latency[4][1] = 40.0  # E - B
    location_latency[4][2] = 40.0  # E - C
    location_latency[4][3] = 25.0  # E - D
    location_latency[4][4] = 0.0  # E - E
    '''

    for l1 in range(NUM_LOCATIONS):
        for l2 in range(NUM_LOCATIONS):
            if l1 == l2:
                location_latency[l1][l2] = 0.0
            else:
                path, weight = dijsktra(graph, location_names[l1], location_names[l2])
                location_latency[l1][l2] = weight


def fillNodeLatencyMatrix(nodelatencyMatrix, NUM_NODES, node_location, delayNodeLocationMatrix):
    # Populate Node Latency Matrix
    for n1 in range(NUM_NODES):
        for n2 in range(NUM_NODES):
            if n1 == n2:
                nodelatencyMatrix[n1][n2] = 0
            else:
                nodelatencyMatrix[n1][n2] = delayNodeLocationMatrix[n1][node_location[n2]] + \
                                            delayNodeLocationMatrix[n2][node_location[n2]]


def fillInstanceMatrix(instance, sfc_instance):
    # a1: s1 s2 s3
    instance[0][0] = 1
    instance[0][1] = 1
    instance[0][2] = 1
    #instance[0][3] = 1

    # a2
    # instance[1][3] = 1;
    # instance[1][4] = 1;
    # instance[1][5] = 1;

    # a3
    # instance[2][6] = 1;
    # instance[2][7] = 1;
    # instance[2][8] = 1;

    # a1: s1 -> s2
    sfc_instance[0][1] = 1
    # s2 -> s3
    sfc_instance[1][2] = 1
    # s3 -> s4
    #sfc_instance[2][3] = 1

    # a2: s4 -> s5
    # sfc_instance[3][4] = 1
    # s5 -> s6
    # sfc_instance[4][5] = 1

    # a3: s7 -> s8
    # sfc_instance[6][7] = 1
    # s8 -> s9
    # sfc_instance[7][8] = 1


def fillCommunicationMatrix(communicationMatrix, NUM_SERVICES, sfcInstance, service_band):
    # Populate Communication Matrix
    for s1 in range(NUM_SERVICES):
        for s2 in range(NUM_SERVICES):
            if sfcInstance[s1][s2] == 1:
                communicationMatrix[s1][s2] = service_band[s2]


def fillUsersApplicationMatrix(user_application, user_cost, NUM_APPLICATIONS, NUM_USER_REQUESTS):
    # Each User uses one 1 random app
    for u in range(NUM_USER_REQUESTS):
        user_application[u][randrange(NUM_APPLICATIONS)] = 1

    # User Cost
    for u in range(NUM_USER_REQUESTS):
        if user_application[u][0] == 1.0:
            user_cost[u] = 0.25  # waste
        ''' 
            elif (user_application[u][1] == 1.0):  
                user_cost[u] = 1.0 # surveillance
            elif (user_application[u][2] == 1.0):  
                user_cost[u] = 0.5 # air
            '''


def fillUserPositions(user_location, xUser, yUser, NUM_USER_REQUESTS):
    # Populate x and y positions of Users - Based on 5 Locations
    for u in range(NUM_USER_REQUESTS):
        if user_location[u] == 0:  # L1
            xUser[u] = randrange(1, 6001, 2)
            yUser[u] = randrange(1, 6001, 2)

        elif user_location[u] == 1:  # L2
            xUser[u] = randrange(6001, 12001, 2)
            yUser[u] = randrange(1, 6001, 2)

        elif user_location[u] == 2:  # L3
            xUser[u] = randrange(12001, 18001, 2)
            yUser[u] = randrange(1, 6001, 2)

        elif user_location[u] == 3:  # L4
            xUser[u] = randrange(1, 6001, 2)
            yUser[u] = randrange(6001, 12001, 2)

        elif user_location[u] == 4:  # L5
            xUser[u] = randrange(6001, 12001, 2)
            yUser[u] = randrange(6001, 12001, 2)

        elif user_location[u] == 5:  # L6
            xUser[u] = randrange(12001, 18001, 2)
            yUser[u] = randrange(6001, 12001, 2)

        elif user_location[u] == 6:  # L7
            xUser[u] = randrange(1, 6001, 2)
            yUser[u] = randrange(12001, 18001, 2)

        elif user_location[u] == 7:  # L8
            xUser[u] = randrange(6001, 12001, 2)
            yUser[u] = randrange(12001, 18001, 2)

        elif user_location[u] == 8:  # L9
            xUser[u] = randrange(12001, 18001, 2)
            yUser[u] = randrange(12001, 18001, 2)


def fillUserLocationLatencyMatrix(user_location_latency, location_latency, user_location, NUM_USER_REQUESTS,
                                  NUM_LOCATIONS):
    # Populate User Location Latency Matrix
    for u in range(NUM_USER_REQUESTS):
        for l in range(NUM_LOCATIONS):
            if user_location[u] == l:  # same location
                user_location_latency[u][l] = 2.0
            else:
                user_location_latency[u][l] = 2.0 + location_latency[l][user_location[u]]


def fillDelayUserNodeMatrix(user_location_latency, delay_node_location, delay_user_node, node_location, NUM_NODES,
                            NUM_USER_REQUESTS):
    for u in range(NUM_USER_REQUESTS):
        for n in range(NUM_NODES):
            delay_user_node[u][n] = user_location_latency[u][node_location[n]] + delay_node_location[n][
                node_location[n]]


def average(lst):
    return sum(lst) / len(lst)


def change_user_locations(sim):
    sim.user_location = []
    fillUserLocation(sim.user_location, sim.NUM_USER_REQUESTS)
    fillUserPositions(sim.user_location, sim.xUser, sim.yUser, sim.NUM_USER_REQUESTS)
    fillUserLocationLatencyMatrix(sim.user_location_latency, sim.location_latency, sim.user_location,
                                  sim.NUM_USER_REQUESTS,
                                  sim.NUM_LOCATIONS)

    fillDelayUserNodeMatrix(sim.user_location_latency, sim.delay_node_location, sim.delay_user_node, sim.node_location,
                            sim.NUM_NODES, sim.NUM_USER_REQUESTS)
