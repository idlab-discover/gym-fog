import cplex
import numpy as np

# Constraints for the MILP model

def add_constraint_sfc_slots_limit(sim, cpx, user_application_id):
    vars = []
    coefs = []
    for a in range(sim.NUM_APPLICATIONS):
        vars.append([])
        coefs.append([])
        for id in range(sim.MAX_IDS):
            vars[a].append([])
            coefs[a].append([])
            for u in range(sim.NUM_USER_REQUESTS):
                vars[a][id].append(user_application_id[u][a][id])
                coefs[a][id].append(sim.user_application[u][a])


    #print(vars)
    #print(coefs)

    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):

            #print(vars[a][id])
            #print(coefs[a][id])

            cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars[a][id], coefs[a][id])],
                                       senses=["L"],
                                       rhs=[sim.NUM_SFC_SLOTS],
                                       names=["c_sfc_slots_limit_" + str(a) + "_" + str(id)])


def add_constraint_users_per_sfc(sim, cpx, user_application_id):
    for u in range(sim.NUM_USER_REQUESTS):
        vars = []
        coefs = []
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                vars.append(user_application_id[u][a][id])
                coefs.append(sim.user_application[u][a])

        cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                   senses=["E"],
                                   rhs=[1.0],
                                   names=["c_users_per_sfc_" + str(u)])


def add_constraint_service_slots_limit(sim, cpx, user_service_association):
    # Cost per User
    vars = []
    coefs = []
    for a in range(sim.NUM_APPLICATIONS):
        vars.append([])
        coefs.append([])
        for id in range(sim.MAX_IDS):
            vars[a].append([])
            coefs[a].append([])
            for s in range(sim.NUM_SERVICES):
                vars[a][id].append([])
                coefs[a][id].append([])
                for rep in range(sim.MAX_REPLICAS):
                    vars[a][id][s].append([])
                    coefs[a][id][s].append([])
                    for n in range(sim.NUM_NODES):
                        vars[a][id][s][rep].append([])
                        coefs[a][id][s][rep].append([])
                        for u in range(sim.NUM_USER_REQUESTS):
                            vars[a][id][s][rep][n].append(user_service_association[u][a][id][s][rep][n])
                            coefs[a][id][s][rep][n].append(float(sim.user_cost[u] * sim.user_application[u][a]))


        #print(vars)
        #print(coefs)

        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                for s in range(sim.NUM_SERVICES):
                    for rep in range(sim.MAX_REPLICAS):
                        for n in range(sim.NUM_NODES):
                            #print(vars[a][id][s][rep][n])
                            #print(coefs[a][id][s][rep][n])
                            limit = sim.service_slots[s]
                            cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars[a][id][s][rep][n], coefs[a][id][s][rep][n])],
                                                       senses=["L"],
                                                       rhs=[limit],
                                                       names=["c_service_slots_limit_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(rep)])


def add_constraint_association_total_services(sim, cpx, user_service_association):
    # User
    # First Part of Equation
    for u in range(sim.NUM_USER_REQUESTS):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                vars = []
                coefs = []
                aux = 0
                for s in range(sim.NUM_SERVICES):
                    aux = aux + int(sim.instance[a][s])
                    for rep in range(sim.MAX_REPLICAS):
                        for n in range(sim.NUM_NODES):
                            vars.append(user_service_association[u][a][id][s][rep][n])
                            coefs.append(1)

                # Second Part of Equation
                # print(aux)
                # print(vars)
                # print(coefs)
                limit = int(aux * sim.user_application[u][a])
                # print(limit)
                cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                           senses=["E"],
                                           rhs=[limit],
                                           names=["c_association_total_services_" + str(u) + "_" + str(a) + "_" + str(id)])


def add_constraint_association_one_per_service(sim, cpx, user_service_association):
    # User
    for u in range(sim.NUM_USER_REQUESTS):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                for s in range(sim.NUM_SERVICES):
                    vars = []
                    coefs = []
                    for rep in range(sim.MAX_REPLICAS):
                        for n in range(sim.NUM_NODES):
                            vars.append(user_service_association[u][a][id][s][rep][n])
                            coefs.append(1)

                    limit = sim.user_application[u][a] * sim.instance[a][s]
                    #print(limit)
                    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                               senses=["E"],
                                               rhs=[limit],
                                               names=["c_association_one_per_service_" + str(u) + "_" + str(a) + "_" + str(
                                                       id) + "_" + str(s)])


def add_constraint_association_service_placement(sim, cpx, user_service_association, placement_matrix):
    # User
    for u in range(sim.NUM_USER_REQUESTS):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                for s in range(sim.NUM_SERVICES):
                    for rep in range(sim.MAX_REPLICAS):
                        for n in range(sim.NUM_NODES):
                            vars = [user_service_association[u][a][id][s][rep][n], placement_matrix[a][id][s][rep][n]]
                            coefs = [1, -1]
                            limit = 0

                            cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                                       senses=["L"],
                                                       rhs=[limit],
                                                       names=["c_association_service_placement_" + str(u) + "_" + str(
                                                           a) + "_" + str(id) + "_" + str(s) + "_" + str(
                                                           rep) + "_" + str(n)])


def add_constraint_services_executed_on_the_network(sim, cpx, placement_matrix, acceptance_matrix):
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                for rep in range(sim.MAX_REPLICAS):
                    for n in range(sim.NUM_NODES):
                        vars = [placement_matrix[a][id][s][rep][n], acceptance_matrix[a][id]]
                        coefs = [1, -1]
                        limit = 0

                        # print(vars)

                        cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                                   senses=["L"],
                                                   rhs=[limit],
                                                   names=["c_services_executed_on_the_network_" + str(a) + "_" + str(
                                                       id) + "_" + str(s) + "_" + str(rep) + "_" + str(n)])


def add_constraint_one_service_request_per_node(sim, cpx, placement_matrix):
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                for rep in range(sim.MAX_REPLICAS):
                    vars = []
                    coefs = []
                    for n in range(sim.NUM_NODES):
                        vars.append(placement_matrix[a][id][s][rep][n])
                        coefs.append(1)

                    # print(vars)
                    # print(coefs)
                    limit = 1
                    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                               senses=["L"],
                                               rhs=[limit],
                                               names=["c_service_request_per_node_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(rep)])


def add_constraint_execution_matrix(sim, cpx, placement_matrix, execution_matrix):
    for s in range(sim.NUM_SERVICES):
        for rep in range(sim.MAX_REPLICAS):
            for n in range(sim.NUM_NODES):
                vars = []
                coefs = []

                prod = 0

                # Equation
                for a in range(sim.NUM_APPLICATIONS):
                    for id in range(sim.MAX_IDS):
                        vars.append(placement_matrix[a][id][s][rep][n])
                        coefs.append(1)
                        prod = prod + (sim.instance[a][s] * sim.MAX_IDS)

                vars.append(execution_matrix[s][n])
                coefs.append(-prod)

                #print(vars)
                #print(coefs)
                #print(prod)

                limit = 0
                cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                           senses=["L"],
                                           rhs=[limit],
                                           names=["c_execution_matrix_" + str(s) + "_" + str(n)])


def add_extra_constraint_objective_min_number_nodes(sim, cpx, execution_matrix, node_utilization_matrix):
    for n in range(sim.NUM_NODES):
        vars = []
        coefs = []

        for s in range(sim.NUM_SERVICES):
            vars.append(execution_matrix[s][n])
            coefs.append(1)

        vars.append(node_utilization_matrix[n])
        coefs.append(-sim.NUM_SERVICES)

        # print(vars)
        # print(coefs)

        limit = 0
        cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                   senses=["L"],
                                   rhs=[limit],
                                   names=["c_objective_min_number_nodes_" + str(n)])


def add_constraint_same_service_different_nodes(sim, cpx, placement_matrix):
    for n in range(sim.NUM_NODES):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                for s in range(sim.NUM_SERVICES):
                    vars = []
                    coefs = []

                    for rep in range(sim.MAX_REPLICAS):
                        vars.append(placement_matrix[a][id][s][rep][n])
                        coefs.append(1)

                    # print(vars)
                    # print(coefs)

                    limit = 1
                    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                               senses=["L"],
                                               rhs=[limit],
                                               names=["c_same_service_different_nodes_" + str(n) + "_" + str(a) + "_" + str(id) + "_" + str(s)])


def add_constraint_service_based_instance_matrix(sim, cpx, placement_matrix):
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                for rep in range(sim.MAX_REPLICAS):
                    vars = []
                    coefs = []

                    for n in range(sim.NUM_NODES):
                        vars.append(placement_matrix[a][id][s][rep][n])
                        coefs.append(1)

                    # print(vars)
                    # print(coefs)

                    limit = sim.instance[a][s]
                    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                               senses=["L"],
                                               rhs=[limit],
                                               names=["c_service_based_instance_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(rep)])


def add_constraint_previous_placement_matrix(sim, cpx, previous_placement_matrix, previous):
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                for rep in range(sim.MAX_REPLICAS):
                    for n in range(sim.NUM_NODES):
                        vars = [previous_placement_matrix[a][id][s][rep][n]]
                        coefs = [1.0]

                        limit = previous[a][id][s][rep][n]
                        #print(limit)

                        cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                                   senses=["E"],
                                                   rhs=[limit],
                                                   names=["c_previous_placement_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(rep) + "_" + str(n)])


def add_constraint_cpu_mem_band_limits(sim, cpx, placement_matrix):
    # CPU MEM BAND
    vars_mem = []
    coefs_mem = []
    vars_cpu = []
    coefs_cpu = []
    vars_band = []
    coefs_band = []
    for n in range(sim.NUM_NODES):
        vars_mem.append([])
        coefs_mem.append([])
        vars_cpu.append([])
        coefs_cpu.append([])
        vars_band.append([])
        coefs_band.append([])
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                for s in range(sim.NUM_SERVICES):
                    for rep in range(sim.MAX_REPLICAS):
                        vars_mem[n].append(placement_matrix[a][id][s][rep][n])
                        coefs_mem[n].append(sim.service_mem[s])
                        vars_cpu[n].append(placement_matrix[a][id][s][rep][n])
                        coefs_cpu[n].append(sim.service_cpu[s])
                        vars_band[n].append(placement_matrix[a][id][s][rep][n])
                        coefs_band[n].append(sim.service_band[s])

        # MEM
        limit = sim.node_mem[n]
        cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars_mem[n], coefs_mem[n])],
                                   senses=["L"],
                                   rhs=[limit],
                                   names=["c_mem_" + str(n)])
        # CPU
        limit = sim.node_cpu[n]
        cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars_cpu[n], coefs_cpu[n])],
                                   senses=["L"],
                                   rhs=[limit],
                                   names=["c_cpu_" + str(n)])
        # band
        limit = sim.node_band[n]
        cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars_band[n], coefs_band[n])],
                                   senses=["L"],
                                   rhs=[limit],
                                   names=["c_band_" + str(n)])


def add_constraint_acceptance_request(sim, cpx, acceptance_matrix, acceptance_service_matrix, placement_matrix,
                                      replicas_per_service):
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            vars_acceptance = []
            coefs_acceptance = []

            sum = 0

            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    vars_acceptance.append(acceptance_service_matrix[a][id][s])
                    coefs_acceptance.append(1)

                    sum = sum + sim.instance[a][s]

                    vars = []
                    coefs = []

                    vars.append(replicas_per_service[a][id][s])
                    coefs.append(sim.instance[a][s])

                    for rep in range(sim.MAX_REPLICAS):
                        for n in range(sim.NUM_NODES):
                            vars.append(placement_matrix[a][id][s][rep][n])
                            coefs.append(-1)

                    limit = 0
                    # print(vars)
                    # print(coefs)
                    # print(limit)

                    # Acceptance Service
                    cpx.indicator_constraints.add(lin_expr=cplex.SparsePair(vars, coefs),
                                                  sense="E",
                                                  rhs=limit,
                                                  indvar=acceptance_service_matrix[a][id][s],
                                                  complemented=0,
                                                  name="ind_c_acceptance_service_matrix_" + str(a) + "_" + str(id) + "_" + str(s),
                                                  indtype=cpx.indicator_constraints.type_.if_)

            # acceptance Matrix -> Equal to 1 means that all Services are accepted!
            cpx.indicator_constraints.add(lin_expr=cplex.SparsePair(vars_acceptance, coefs_acceptance),
                                          sense="E",
                                          rhs=sum,
                                          indvar=acceptance_matrix[a][id],
                                          complemented=0,
                                          name="ind_c_acceptance_matrix_" + str(a) + "_" + str(id),
                                          indtype=cpx.indicator_constraints.type_.if_)


def add_constraint_user_delay(sim, cpx, user_delay, user_service_association):
    for u in range(sim.NUM_USER_REQUESTS):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                for s in range(sim.NUM_SERVICES):
                    if (sim.instance[a][s] == 1) and (sim.service_sfc_last_pos[s]):
                        for rep in range(sim.MAX_REPLICAS):
                            for n in range(sim.NUM_NODES):
                                vars_eq1 = []
                                coefs_eq1 = []

                                # User Delay
                                vars_eq1.append(user_delay[u])
                                coefs_eq1.append(1.0)

                                #print(vars_eq1)
                                #print(coefs_eq1)

                                # User Latency
                                cpx.indicator_constraints.add(
                                    lin_expr=cplex.SparsePair(vars_eq1, coefs_eq1),
                                    sense="E",
                                    rhs=sim.delay_user_node[u][n],
                                    indvar=user_service_association[u][a][id][s][rep][n],
                                    complemented=0,
                                    name="ind_c_user_latency_" + str(u) + "_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(rep) + "_" + str(n),
                                    indtype=cpx.indicator_constraints.type_.if_)


def add_constraint_migrations(sim, cpx, migrations_matrix, previous_placement_matrix, placement_matrix):
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    for rep in range(sim.MAX_REPLICAS):
                        for n in range(sim.NUM_NODES):
                            vars = []
                            coefs = []

                            vars_2 = []
                            coefs_2 = []

                            vars.append(previous_placement_matrix[a][id][s][rep][n])
                            coefs.append(1.0)

                            vars.append(placement_matrix[a][id][s][rep][n])
                            coefs.append(1.0)

                            vars_2.append(previous_placement_matrix[a][id][s][rep][n])
                            coefs_2.append(1.0)

                            vars_2.append(placement_matrix[a][id][s][rep][n])
                            coefs_2.append(-1.0)

                            #print(vars)
                            #print(coefs)

                            # Migrations Equal 1 if Previous + Placement = 1
                            cpx.indicator_constraints.add(
                                lin_expr=cplex.SparsePair(vars, coefs),
                                sense="E",
                                rhs=1.0,
                                indvar=migrations_matrix[a][id][s][rep][n],
                                complemented=0,
                                name="ind_c_migrations_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(rep) + "_" + str(n),
                                indtype=cpx.indicator_constraints.type_.if_)

                            # Migrations Equal 0 if Previous - Placement = 0
                            cpx.indicator_constraints.add(
                                lin_expr=cplex.SparsePair(vars, coefs),
                                sense="E",
                                rhs=0,
                                indvar=migrations_matrix[a][id][s][rep][n],
                                complemented=1,
                                name="ind_c_migrations_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(
                                    rep) + "_" + str(n),
                                indtype=cpx.indicator_constraints.type_.if_)


def add_constraint_migrations_2nd(sim, cpx, migrations_matrix, service_migrations_matrix):
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    for rep in range(sim.MAX_REPLICAS):
                        vars = []
                        coefs = []

                        for n in range(sim.NUM_NODES):
                            vars.append(migrations_matrix[a][id][s][rep][n])
                            coefs.append(1.0)

                        # Service Migrations Equal 1 if sum (Migrations) >= 1
                        cpx.indicator_constraints.add(
                            lin_expr=cplex.SparsePair(vars, coefs),
                            sense="G",
                            rhs=1,
                            indvar=service_migrations_matrix[a][id][s][rep],
                            complemented=0,
                            name="ind_c_service_migrations_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(rep),
                            indtype=cpx.indicator_constraints.type_.if_)

                        # Service Migrations Equal 0 if sum (Migrations) = 0
                        cpx.indicator_constraints.add(
                            lin_expr=cplex.SparsePair(vars, coefs),
                            sense="E",
                            rhs=0,
                            indvar=service_migrations_matrix[a][id][s][rep],
                            complemented=1,
                            name="ind_c_service_migrations_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(
                                rep),
                            indtype=cpx.indicator_constraints.type_.if_)


def add_constraint_restrict_migrations(sim, cpx, replicas_per_service, service_migrations_matrix):
        vars = []
        coefs = []

        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):

                for s in range(sim.NUM_SERVICES):
                    if sim.instance[a][s] == 1:
                        vars.append(replicas_per_service[a][id][s])
                        coefs.append(-sim.MIGRATION_FACTOR)

                        for rep in range(sim.MAX_REPLICAS):
                            vars.append(service_migrations_matrix[a][id][s][rep])
                            coefs.append(1)

        print(vars)
        print(coefs)

        # Restrict Migrations: 22 * 30% = 6.6 => 6
        cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                                   senses=["L"],
                                   rhs=[0],
                                   names=["c_restrict_migrations"])


def add_constraint_objective_max_acceptance_user_requests(sim, cpx, user_application_id, value_obj):
    vars = []
    coefs = []
    for u in range(sim.NUM_USER_REQUESTS):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                vars.append(user_application_id[u][a][id])
                coefs.append(sim.user_application[u][a])

    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                               senses=["E"],
                               rhs=[value_obj],
                               names=["c_objective_max_user_requests"])


def add_constraint_objective_number_service_replicas(sim, cpx, replicas_per_service, value_obj):
    vars = []
    coefs = []
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                vars.append(replicas_per_service[a][id][s])
                coefs.append(1.0)

    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                               senses=["E"],
                               rhs=[value_obj],
                               names=["c_objective_number_service_replicas"])


def add_constraint_objective_min_nodes(sim, cpx, node_utilization_matrix, value_obj):
    vars = []
    coefs = []
    for n in range(sim.NUM_NODES):
        vars.append(node_utilization_matrix[n])
        coefs.append(1.0)

    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                               senses=["E"],
                               rhs=[value_obj],
                               names=["c_objective_min_number_nodes"])


def add_constraint_objective_min_cost(sim, cpx, placement_matrix, value_obj):
    vars = []
    coefs = []
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                for rep in range(sim.MAX_REPLICAS):
                    for n in range(sim.NUM_NODES):
                        vars.append(placement_matrix[a][id][s][rep][n])
                        coefs.append(1.0 * sim.weigh_node[n] * sim.service_cpu[s] * sim.service_mem[s] * sim.service_band[s])

    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                               senses=["E"],
                               rhs=[value_obj],
                               names=["c_objective_min_cost"])


def add_constraint_objective_min_user_delay(sim, cpx, user_delay, value_obj):
    vars = []
    coefs = []
    for u in range(sim.NUM_USER_REQUESTS):
        vars.append(user_delay[u])
        coefs.append(1.0)

    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(vars, coefs)],
                               senses=["E"],
                               rhs=[value_obj],
                               names=["c_objective_min_user_delay"])