import logging
import random as rand
from gym_fog.milp.util import average


def getPercentageRequests(s1_rl, s2_rl, s3_rl, s1_milp, s2_milp, s3_milp, number_users):
    # Extra Large
    if number_users <= 16:  # 1 1 1
        if s1_rl >= s1_milp and s2_rl >= s2_milp and s3_rl >= s3_milp:
            return 100
        else:
            return 0
    elif number_users <= 24:  # 2 1 2
        if s1_rl >= s1_milp and s2_rl >= s2_milp and s3_rl >= s3_milp:
            return 100
        elif s1_rl >= 1 and s2_rl >= 1 and s3_rl >= 1:
            return 100 - (((number_users - 16) / number_users) * 100)
        else:
            return 0
    elif number_users <= 32:  # 2 2 2
        if s1_rl >= s1_milp and s2_rl >= s2_milp and s3_rl >= s3_milp:
            return 100
        elif s1_rl >= 2 and s2_rl >= 1 and s3_rl >= 2:
            return 100 - (((number_users - 24) / number_users) * 100)
        elif s1_rl >= 1 and s2_rl >= 1 and s3_rl >= 1:
            return 100 - (((number_users - 16) / number_users) * 100)
        else:
            return 0
    elif number_users <= 48:  # 3 2 3
        if s1_rl >= s1_milp and s2_rl >= s2_milp and s3_rl >= s3_milp:
            return 100
        elif s1_rl >= 2 and s2_rl >= 2 and s3_rl >= 2:
            return 100 - (((number_users - 32) / number_users) * 100)
        elif s1_rl >= 2 and s2_rl >= 1 and s3_rl >= 2:
            return 100 - (((number_users - 24) / number_users) * 100)
        elif s1_rl >= 1 and s2_rl >= 1 and s3_rl >= 1:
            return 100 - (((number_users - 16) / number_users) * 100)
        else:
            return 0
    else:  # >48 # 4 3 4
        if s1_rl >= s1_milp and s2_rl >= s2_milp and s3_rl >= s3_milp:
            return 100
        elif s1_rl >= 3 and s2_rl >= 2 and s3_rl >= 3:
            return 100 - (((number_users - 48) / number_users) * 100)
        elif s1_rl >= 2 and s2_rl >= 2 and s3_rl >= 2:
            return 100 - (((number_users - 32) / number_users) * 100)
        elif s1_rl >= 2 and s2_rl >= 1 and s3_rl >= 2:
            return 100 - (((number_users - 24) / number_users) * 100)
        elif s1_rl >= 1 and s2_rl >= 1 and s3_rl >= 1:
            return 100 - (((number_users - 16) / number_users) * 100)
        else:
            return 0


def get_replicas(a, i, s, fog_env):
    replicas = []
    for rep in range(fog_env.simulation.MAX_REPLICAS):
        # If the replica is not deployed
        if not fog_env.rl_placement_matrix[a][i][s][rep].any():
            replicas.append(rep)

    # logging.info("Available Replicas: " + str(replicas))
    return replicas


def reset_node_constraints(fog_env):
    for n in range(fog_env.simulation.NUM_NODES):
        fog_env.rl_node_av_cpu[n] = fog_env.simulation.node_cpu[n]
        fog_env.rl_node_av_mem[n] = fog_env.simulation.node_mem[n]
        fog_env.rl_node_av_band[n] = fog_env.simulation.node_band[n]


def get_replica(a, i, s, n, fog_env):
    replica = fog_env.simulation.MAX_REPLICAS + 1

    for rep in range(fog_env.simulation.MAX_REPLICAS):
        # If the replica is deployed
        if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
            replica = rep

    # logging.info("Replica: " + str(replica))
    return replica


def get_service_replica_node(a, i, s, fog_env):
    replica = fog_env.simulation.MAX_REPLICAS + 1
    node = fog_env.simulation.NUM_NODES + 1

    for rep in range(fog_env.simulation.MAX_REPLICAS):
        for n in range(fog_env.simulation.NUM_NODES):
            # If the replica is deployed on that node
            if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
                replica = rep
                node = n

    return replica, node


def check_service(a, i, s, n, fog_env):
    for rep in range(fog_env.simulation.MAX_REPLICAS):
        # If the replica is deployed
        if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
            return False

    return True


def check_capacity(s, n, fog_env):
    # return True

    if fog_env.rl_node_av_cpu[n] >= fog_env.simulation.service_cpu[s] \
            and fog_env.rl_node_av_mem[n] >= fog_env.simulation.service_mem[s] \
            and fog_env.rl_node_av_band[n] >= fog_env.simulation.service_band[s]:
        return True
    else:
        return False


def check_user(u, a, i, s, fog_env):
    for rep in range(fog_env.simulation.MAX_REPLICAS):
        # If the user is already associated
        if fog_env.rl_user_service_association[u][a][i][s][rep].any():
            return False

    return True


def check_user_node(u, a, i, s, n, fog_env):
    if u > (fog_env.simulation.NUM_USER_REQUESTS - 1):
        return False

    for rep in range(fog_env.simulation.MAX_REPLICAS):
        if fog_env.rl_user_service_association[u][a][i][s][rep][n] == 1:
            return True

    return False


def user_node_observation(u, n, fog_env):
    if check_user_node(u, 0, 0, 2, n, fog_env):
        return 1
    else:
        return 0


def deploy_service(a, i, s, n, fog_env):
    """Deploy the service if possible"""

    # Check Node Available Capacity
    if check_capacity(s, n, fog_env):

        # Only one Service per node!
        if check_service(a, i, s, n, fog_env):
            available_replicas = get_replicas(a, i, s, fog_env)
            if not available_replicas:
                # logging.info("Constraint: Add Service MAX Replicas Reached!")
                fog_env.constraint_add_service_max_replicas_reached = True

            else:
                random_row = rand.randrange(len(available_replicas))

                rep = available_replicas[random_row]

                # Deploy Service
                fog_env.rl_placement_matrix[a][i][s][rep][n] = 1

                # decrease capacity of Node
                fog_env.rl_node_av_cpu[n] -= fog_env.simulation.service_cpu[s]
                fog_env.rl_node_av_mem[n] -= fog_env.simulation.service_mem[s]
                fog_env.rl_node_av_band[n] -= fog_env.simulation.service_band[s]

                # logging.info("Service Deployed: P[a" + str(a + 1) + "][id" + str(i + 1) + "][s" + str(s + 1) + "][rep " + str(rep + 1) + "][n" + str(n + 1) + "]")
                # Reward Keyword
                fog_env.deploy_service = True
            return
        else:
            # logging.info("Constraint: MAX Number of Services per node reached!")

            # Reward Keyword
            fog_env.constraint_max_services_on_node = True
            return
    else:
        logging.info("Constraint: MAX Node Capacity reached!")

        # Reward Keyword
        fog_env.constraint_node_capacity = True
        return


def terminate_service(a, i, s, n, fog_env):
    """Terminate the service if possible"""
    rep = get_replica(a, i, s, n, fog_env)
    if rep > fog_env.simulation.MAX_REPLICAS:
        # logging.info("Constraint: Terminate Service without deployed Service!")
        fog_env.constraint_terminate_service_without_deployed_services = True

    else:
        # Terminate Service
        fog_env.rl_placement_matrix[a][i][s][rep][n] = 0

        # Remove all User Associations:
        remove_users_from_service_association(a, i, s, rep, n, fog_env)

        # add free up capacity to Node
        fog_env.rl_node_av_cpu[n] += fog_env.simulation.service_cpu[s]
        fog_env.rl_node_av_mem[n] += fog_env.simulation.service_mem[s]
        fog_env.rl_node_av_band[n] += fog_env.simulation.service_band[s]

        # Reward Keyword
        fog_env.terminate_service = True


def add_user(u, a, i, s, n, fog_env):
    """Add User to Specific service if possible"""
    if u > (fog_env.simulation.NUM_USER_REQUESTS - 1):
        fog_env.constraint_user_request_absent = True
        return

    # Only one User per the same Service!
    if check_user(u, a, i, s, fog_env):
        # Get specific replica!
        rep = get_replica(a, i, s, n, fog_env)

        if rep > fog_env.simulation.MAX_REPLICAS:
            # logging.info("No service allocated, cannot add user to service replica!")
            fog_env.constraint_add_user_without_deployed_services = True

        else:
            # Add User
            fog_env.rl_user_service_association[u][a][i][s][rep][n] = 1

            # Update User Delay if service is last of chain
            if fog_env.simulation.service_sfc_last_pos[s] == 1:
                fog_env.rl_user_delay[u] = fog_env.simulation.delay_user_node[u][n]
                logging.info("User " + str(u) + " Delay updated: " + str(fog_env.rl_user_delay[u]))

            # Reward Keyword
            fog_env.add_user = True
    else:
        # logging.info("User Already allocated to this Service! Remove it First!")
        # Reward Keyword
        fog_env.constraint_user_per_service_instance = True
        return


def remove_user(u, a, i, s, fog_env):
    """Remove User from Specific service if possible"""
    if u > (fog_env.simulation.NUM_USER_REQUESTS - 1):
        fog_env.constraint_user_request_absent = True
        return

    # Get specific replica!
    rep, n = get_service_replica_node(a, i, s, fog_env)

    if rep > fog_env.simulation.MAX_REPLICAS and n > fog_env.simulation.NUM_NODES:
        # logging.info("No service allocated, cannot remove user from service replica!")
        fog_env.constraint_remove_user_without_service_allocated = True

    else:
        if fog_env.rl_user_service_association[u][a][i][s][rep][n] == 1:
            # Remove User
            fog_env.rl_user_service_association[u][a][i][s][rep][n] = 0

            # Update User Delay if service is last of chain
            if fog_env.simulation.service_sfc_last_pos[s] == 1:
                fog_env.rl_user_delay[u] = fog_env.max_user_delay
                logging.info("User removed " + str(u) + " Delay updated: " + str(fog_env.rl_user_delay[u]))

            # Reward Keyword
            fog_env.remove_user = True
        else:
            # logging.info("User is not associated!, cannot remove user!")
            fog_env.constraint_remove_user_without_user_associated = True


def remove_users_from_service_association(a, i, s, rep, n, fog_env):
    for u in range(fog_env.simulation.NUM_USER_REQUESTS):
        if fog_env.rl_user_service_association[u][a][i][s][rep][n] == 1:
            fog_env.rl_user_service_association[u][a][i][s][rep][n] = 0
            fog_env.rl_user_delay[u] = fog_env.max_user_delay
            logging.info("Service Terminated! User " + str(u) + " Delay updated: " + str(fog_env.rl_user_delay[u]))

    return


def get_ratio_user_allocated(fog_env):
    milp = 0
    rl = 0
    for u in range(fog_env.simulation.NUM_USER_REQUESTS):
        for a in range(fog_env.simulation.NUM_APPLICATIONS):
            # User preferred application
            if fog_env.simulation.user_application[u][a] == 1:
                for i in range(fog_env.simulation.MAX_IDS):
                    for s in range(fog_env.simulation.NUM_SERVICES):
                        # If the service belongs to the app
                        if fog_env.simulation.instance[a][s] == 1:
                            for rep in range(fog_env.simulation.MAX_REPLICAS):
                                for n in range(fog_env.simulation.NUM_NODES):
                                    if fog_env.simulation.milp_solution_user_service_association[u][a][i][s][rep][
                                        n] == 1:
                                        milp += 1
                                    if fog_env.rl_user_service_association[u][a][i][s][rep][n] == 1:
                                        rl += 1
    if milp == 0:
        ratio = 0
    else:
        ratio = rl / milp

    return ratio


def get_ratio_total_services(fog_env):
    milp = 0
    rl = 0
    for a in range(fog_env.simulation.NUM_APPLICATIONS):
        for i in range(fog_env.simulation.MAX_IDS):
            for s in range(fog_env.simulation.NUM_SERVICES):
                for rep in range(fog_env.simulation.MAX_REPLICAS):
                    for n in range(fog_env.simulation.NUM_NODES):
                        if fog_env.simulation.milp_solution_placement_matrix[a][i][s][rep][n] == 1:
                            milp += 1
                        if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
                            rl += 1

    if milp == 0:
        ratio = 0
    else:
        ratio = rl / milp

    return ratio


def get_min_cost_milp(fog_env):
    cost_milp = 0

    for a in range(fog_env.simulation.NUM_APPLICATIONS):
        for i in range(fog_env.simulation.MAX_IDS):
            for s in range(fog_env.simulation.NUM_SERVICES):
                for rep in range(fog_env.simulation.MAX_REPLICAS):
                    for n in range(fog_env.simulation.NUM_NODES):
                        if fog_env.simulation.milp_solution_placement_matrix[a][i][s][rep][n] == 1:
                            cost_milp += 1.0 * fog_env.simulation.weigh_node[n] * fog_env.simulation.service_cpu[s] * \
                                         fog_env.simulation.service_mem[s] * fog_env.simulation.service_band[s]

    # logging.info("Cost MILP: " + str(cost_milp))

    return cost_milp


def get_min_cost_rl(fog_env):
    cost_rl = 0

    for a in range(fog_env.simulation.NUM_APPLICATIONS):
        for i in range(fog_env.simulation.MAX_IDS):
            for s in range(fog_env.simulation.NUM_SERVICES):
                for rep in range(fog_env.simulation.MAX_REPLICAS):
                    for n in range(fog_env.simulation.NUM_NODES):
                        if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
                            cost_rl += 1.0 * fog_env.simulation.weigh_node[n] * fog_env.simulation.service_cpu[s] * \
                                       fog_env.simulation.service_mem[s] * fog_env.simulation.service_band[s]

    # logging.info("Cost RL: " + str(cost_rl))
    return cost_rl


def get_min_cost_diff(fog_env):
    diff = get_min_cost_rl(fog_env) - get_min_cost_milp(fog_env)
    # logging.info("Diff: " + str(diff))

    return diff


def get_lat_percentage(fog_env):
    lat_rl = get_user_delay_rl(fog_env)
    lat_milp = get_user_delay_milp(fog_env)

    percentage = ((lat_rl - lat_milp) / lat_milp) * 100
    return percentage


def get_min_cost_percentage(fog_env):
    cost_rl = get_min_cost_rl(fog_env)
    cost_milp = get_min_cost_milp(fog_env)

    percentage = ((cost_rl - cost_milp) / cost_milp) * 100

    # logging.info("Percentage: " + str(percentage))
    return percentage


def get_user_ratio_service(s, fog_env):
    milp = 0
    rl = 0
    for u in range(fog_env.simulation.NUM_USER_REQUESTS):
        for a in range(fog_env.simulation.NUM_APPLICATIONS):
            for i in range(fog_env.simulation.MAX_IDS):
                for rep in range(fog_env.simulation.MAX_REPLICAS):
                    for n in range(fog_env.simulation.NUM_NODES):
                        if fog_env.simulation.milp_solution_user_service_association[u][a][i][s][rep][n] == 1:
                            milp += 1
                        if fog_env.rl_user_service_association[u][a][i][s][rep][n] == 1:
                            rl += 1

    if milp == 0:
        ratio = 0
    else:
        ratio = rl / milp

    return ratio


def get_number_deployed_service_instances_on_node(s, n, fog_env):
    rl = 0

    for a in range(fog_env.simulation.NUM_APPLICATIONS):
        for i in range(fog_env.simulation.MAX_IDS):
            for rep in range(fog_env.simulation.MAX_REPLICAS):
                if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
                    rl += 1

    return rl


def get_number_deployed_service_instances(s, fog_env):
    rl = 0

    for a in range(fog_env.simulation.NUM_APPLICATIONS):
        for i in range(fog_env.simulation.MAX_IDS):
            for rep in range(fog_env.simulation.MAX_REPLICAS):
                for n in range(fog_env.simulation.NUM_NODES):
                    if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
                        rl += 1

    return rl


def get_service_RL(s, fog_env):
    rl = 0
    for a in range(fog_env.simulation.NUM_APPLICATIONS):
        for i in range(fog_env.simulation.MAX_IDS):
            for rep in range(fog_env.simulation.MAX_REPLICAS):
                for n in range(fog_env.simulation.NUM_NODES):
                    if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
                        rl += 1

    return rl


def get_service_MILP(s, fog_env):
    milp = 0
    for a in range(fog_env.simulation.NUM_APPLICATIONS):
        for i in range(fog_env.simulation.MAX_IDS):
            for rep in range(fog_env.simulation.MAX_REPLICAS):
                for n in range(fog_env.simulation.NUM_NODES):
                    if fog_env.simulation.milp_solution_placement_matrix[a][i][s][rep][n] == 1:
                        milp += 1

    return milp


def get_ratio_service(s, fog_env):
    milp = 0
    rl = 0

    for a in range(fog_env.simulation.NUM_APPLICATIONS):
        for i in range(fog_env.simulation.MAX_IDS):
            for rep in range(fog_env.simulation.MAX_REPLICAS):
                for n in range(fog_env.simulation.NUM_NODES):
                    if fog_env.simulation.milp_solution_placement_matrix[a][i][s][rep][n] == 1:
                        milp += 1
                    if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
                        rl += 1

    if milp == 0:
        ratio = 0
    else:
        ratio = float(rl / milp)

    return ratio


def get_cost_service_diff(s, fog_env):
    milp = 0
    rl = 0

    for a in range(fog_env.simulation.NUM_APPLICATIONS):
        for i in range(fog_env.simulation.MAX_IDS):
            for rep in range(fog_env.simulation.MAX_REPLICAS):
                for n in range(fog_env.simulation.NUM_NODES):
                    if fog_env.simulation.milp_solution_placement_matrix[a][i][s][rep][n] == 1:
                        milp += 1
                    if fog_env.rl_placement_matrix[a][i][s][rep][n] == 1:
                        rl += 1

    diff = rl - milp
    return diff


def get_user_delay_rl(fog_env):
    delay_rl = []
    for u in range(fog_env.simulation.NUM_USER_REQUESTS):
        if fog_env.rl_user_delay[u] == 0.0:
            delay_rl.append(fog_env.MAX_OBSERVATION_SPACE)
        else:
            delay_rl.append(fog_env.rl_user_delay[u])

    return average(delay_rl)


def get_user_delay_milp(fog_env):
    delay_milp = []
    for u in range(fog_env.simulation.NUM_USER_REQUESTS):
        delay_milp.append(fog_env.simulation.milp_solution_user_delay[u])

    for value in delay_milp:
        if value == 0.0:
            return 0.0

    return average(delay_milp)


def change_user_rl_variables(fog_env, rl_user_delay, rl_user_service_association, previous_number_users):
    # previous solutions
    for u in range(previous_number_users):
        fog_env.rl_user_delay[u] = rl_user_delay[u]

    for u in range(previous_number_users):
        for a in range(fog_env.simulation.NUM_APPLICATIONS):
            for i in range(fog_env.simulation.MAX_IDS):
                for s in range(fog_env.simulation.NUM_SERVICES):
                    for rep in range(fog_env.simulation.MAX_REPLICAS):
                        for n in range(fog_env.simulation.NUM_NODES):
                            fog_env.rl_user_service_association[u][a][i][s][rep][n] = \
                                rl_user_service_association[u][a][i][s][rep][n]

    return
