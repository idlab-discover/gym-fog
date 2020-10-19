def objective_MAX_acceptance_user_requests(sim, user_application_id):
    vars = []
    coefs = []
    for u in range(sim.NUM_USER_REQUESTS):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                vars.append(user_application_id[u][a][id])
                coefs.append(sim.user_application[u][a])

    return zip(vars, coefs)


def objective_number_service_replicas(sim, replicas_per_service):
    vars = []
    coefs = []
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                vars.append(replicas_per_service[a][id][s])
                coefs.append(1.0)

    return zip(vars, coefs)


def objective_MIN_number_nodes(sim, node_utilization_matrix):
    vars = []
    coefs = []
    for n in range(sim.NUM_NODES):
        vars.append(node_utilization_matrix[n])
        coefs.append(1.0)

    return zip(vars, coefs)


def objective_MIN_cost(sim, placement_matrix):
    vars = []
    coefs = []
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                for rep in range(sim.MAX_REPLICAS):
                    for n in range(sim.NUM_NODES):
                        vars.append(placement_matrix[a][id][s][rep][n])
                        coefs.append(1.0 * sim.weigh_node[n] * sim.service_cpu[s] * sim.service_mem[s] * sim.service_band[s])

    return zip(vars, coefs)


def objective_MIN_user_access_delay(sim, user_delay):
    vars = []
    coefs = []
    for u in range(sim.NUM_USER_REQUESTS):
        vars.append(user_delay[u])
        coefs.append(1.0)

    return zip(vars, coefs)
