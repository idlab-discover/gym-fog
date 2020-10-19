import logging

import cplex
import numpy as np

from gym_fog.milp.constraint import add_constraint_sfc_slots_limit, add_constraint_users_per_sfc, \
    add_constraint_service_slots_limit, add_constraint_association_total_services, \
    add_constraint_association_one_per_service, add_constraint_association_service_placement, \
    add_constraint_services_executed_on_the_network, add_constraint_one_service_request_per_node, \
    add_constraint_execution_matrix, add_extra_constraint_objective_min_number_nodes, \
    add_constraint_same_service_different_nodes, add_constraint_service_based_instance_matrix, \
    add_constraint_previous_placement_matrix, add_constraint_cpu_mem_band_limits, add_constraint_acceptance_request, \
    add_constraint_user_delay, add_constraint_migrations, add_constraint_migrations_2nd, \
    add_constraint_objective_max_acceptance_user_requests, add_constraint_objective_number_service_replicas, \
    add_constraint_objective_min_nodes, add_constraint_objective_min_user_delay, add_constraint_objective_min_cost
from gym_fog.milp.optimization import objective_MAX_acceptance_user_requests, objective_MIN_number_nodes, \
    objective_MIN_user_access_delay, objective_number_service_replicas, objective_MIN_cost

# CPLEX VARIABLES
all_acceptance_matrix = []
acceptance_matrix = []
all_acceptance_service_matrix = []
acceptance_service_matrix = []
all_execution_matrix = []
execution_matrix = []
all_replicas_per_service = []
replicas_per_service = []
all_placement_matrix = []
placement_matrix = []
all_previous_placement_matrix = []
previous_placement_matrix = []
node_utilization_matrix = []
all_migrations_matrix = []
migrations_matrix = []
all_service_migrations_matrix = []
service_migrations_matrix = []
all_user_application_id = []
user_application_id = []
all_user_service_association = []
user_service_association = []
user_delay = []

def reset_variables():
    all_acceptance_matrix.clear()
    acceptance_matrix.clear()
    all_acceptance_service_matrix.clear()
    acceptance_service_matrix.clear()
    all_execution_matrix.clear()
    execution_matrix.clear()
    all_replicas_per_service.clear()
    replicas_per_service.clear()
    all_placement_matrix.clear()
    placement_matrix.clear()
    all_previous_placement_matrix.clear()
    previous_placement_matrix.clear()
    node_utilization_matrix.clear()
    all_migrations_matrix.clear()
    migrations_matrix.clear()
    all_service_migrations_matrix.clear()
    service_migrations_matrix.clear()
    all_user_application_id.clear()
    user_application_id.clear()
    all_user_service_association.clear()
    user_service_association.clear()
    user_delay.clear()

def initializa_variables(sim, cpx):

    # 1) acceptanceMatrix[a][id] -> binary
    for a in range(sim.NUM_APPLICATIONS):
        acceptance_matrix.append([])
        for id in range(sim.MAX_IDS):
            varname = "acceptance_matrix_" + str(a) + "_" + str(id)
            all_acceptance_matrix.append(varname)
            acceptance_matrix[a].append(varname)

    # print(acceptance_matrix)
    # print(all_acceptance_matrix)

    cpx.variables.add(names=all_acceptance_matrix,
                      lb=[0] * len(all_acceptance_matrix),
                      ub=[1] * len(all_acceptance_matrix),
                      types=["B"] * len(all_acceptance_matrix))

    # 2) acceptanceServiceMatrix[a][id][s] -> binary
    for a in range(sim.NUM_APPLICATIONS):
        acceptance_service_matrix.append([])
        for id in range(sim.MAX_IDS):
            acceptance_service_matrix[a].append([])
            for s in range(sim.NUM_SERVICES):
                varname = "acceptance_service_matrix_" + str(a) + "_" + str(id) + "_" + str(s)
                all_acceptance_service_matrix.append(varname)
                acceptance_service_matrix[a][id].append(varname)

    # print(acceptance_service_matrix)
    # print(all_acceptance_service_matrix)

    cpx.variables.add(names=all_acceptance_service_matrix,
                      lb=[0] * len(all_acceptance_service_matrix),
                      ub=[1] * len(all_acceptance_service_matrix),
                      types=["B"] * len(all_acceptance_service_matrix))

    # 3) execution_matrix[s][n] -> binary
    for s in range(sim.NUM_SERVICES):
        execution_matrix.append([])
        for n in range(sim.NUM_NODES):
            varname = "execution_matrix_" + str(s) + "_" + str(n)
            all_execution_matrix.append(varname)
            execution_matrix[s].append(varname)

    # print(execution_matrix)
    # print(all_execution_matrix)

    cpx.variables.add(names=all_execution_matrix,
                      lb=[0] * len(all_execution_matrix),
                      ub=[1] * len(all_execution_matrix),
                      types=["B"] * len(all_execution_matrix))

    # 4) replicas_per_service[a][id][s] -> numerical
    for a in range(sim.NUM_APPLICATIONS):
        replicas_per_service.append([])
        for id in range(sim.MAX_IDS):
            replicas_per_service[a].append([])
            for s in range(sim.NUM_SERVICES):
                varname = "replicas_per_service_" + str(a) + "_" + str(id) + "_" + str(s)
                all_replicas_per_service.append(varname)
                replicas_per_service[a][id].append(varname)

    # print(all_replicas_per_service)
    # print(replicas_per_service)

    cpx.variables.add(names=all_replicas_per_service,
                      lb=[0] * len(all_replicas_per_service),
                      ub=[sim.MAX_REPLICAS] * len(all_replicas_per_service),
                      types=["N"] * len(all_replicas_per_service))

    # 5) placement_matrix[a][id][s][rep][n] -> binary
    for a in range(sim.NUM_APPLICATIONS):
        placement_matrix.append([])
        for id in range(sim.MAX_IDS):
            placement_matrix[a].append([])
            for s in range(sim.NUM_SERVICES):
                placement_matrix[a][id].append([])
                for rep in range(sim.MAX_REPLICAS):
                    placement_matrix[a][id][s].append([])
                    for n in range(sim.NUM_NODES):
                        varname = "placement_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(
                            rep) + "_" + str(n)
                        all_placement_matrix.append(varname)
                        placement_matrix[a][id][s][rep].append(varname)

    # print(all_placement_matrix)
    # print(placement_matrix)

    cpx.variables.add(names=all_placement_matrix,
                      lb=[0] * len(all_placement_matrix),
                      ub=[1] * len(all_placement_matrix),
                      types=["B"] * len(all_placement_matrix))

    # 6) previous_placement_matrix[a][id][s][rep][n] -> binary
    for a in range(sim.NUM_APPLICATIONS):
        previous_placement_matrix.append([])
        for id in range(sim.MAX_IDS):
            previous_placement_matrix[a].append([])
            for s in range(sim.NUM_SERVICES):
                previous_placement_matrix[a][id].append([])
                for rep in range(sim.MAX_REPLICAS):
                    previous_placement_matrix[a][id][s].append([])
                    for n in range(sim.NUM_NODES):
                        varname = "previous_placement_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(
                            rep) + "_" + str(n)
                        all_previous_placement_matrix.append(varname)
                        previous_placement_matrix[a][id][s][rep].append(varname)

    # print(all_previous_placement_matrix)
    # print(previous_placement_matrix)

    cpx.variables.add(names=all_previous_placement_matrix,
                      lb=[0] * len(all_previous_placement_matrix),
                      ub=[1] * len(all_previous_placement_matrix),
                      types=["B"] * len(all_previous_placement_matrix))

    # 7) node_utilization_matrix[n] -> binary
    for n in range(sim.NUM_NODES):
        varname = "node_utilization_matrix_" + str(n)
        node_utilization_matrix.append(varname)

    # print(node_utilization_matrix)

    cpx.variables.add(names=node_utilization_matrix,
                      lb=[0] * len(node_utilization_matrix),
                      ub=[1] * len(node_utilization_matrix),
                      types=["B"] * len(node_utilization_matrix))

    # 8) migrations_matrix[a][id][s][rep][n] -> binary
    for a in range(sim.NUM_APPLICATIONS):
        migrations_matrix.append([])
        for id in range(sim.MAX_IDS):
            migrations_matrix[a].append([])
            for s in range(sim.NUM_SERVICES):
                migrations_matrix[a][id].append([])
                for rep in range(sim.MAX_REPLICAS):
                    migrations_matrix[a][id][s].append([])
                    for n in range(sim.NUM_NODES):
                        varname = "migrations_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(
                            rep) + "_" + str(n)
                        all_migrations_matrix.append(varname)
                        migrations_matrix[a][id][s][rep].append(varname)

    # print(all_migrations_matrix)

    cpx.variables.add(names=all_migrations_matrix,
                      lb=[0] * len(all_migrations_matrix),
                      ub=[1] * len(all_migrations_matrix),
                      types=["B"] * len(all_migrations_matrix))

    # 9) service_migrations_matrix[a][id][s][rep] -> binary
    for a in range(sim.NUM_APPLICATIONS):
        service_migrations_matrix.append([])
        for id in range(sim.MAX_IDS):
            service_migrations_matrix[a].append([])
            for s in range(sim.NUM_SERVICES):
                service_migrations_matrix[a][id].append([])
                for rep in range(sim.MAX_REPLICAS):
                    varname = "service_migrations_matrix_" + str(a) + "_" + str(id) + "_" + str(s) + "_" + str(rep)
                    all_service_migrations_matrix.append(varname)
                    service_migrations_matrix[a][id][s].append(varname)

    # print(all_service_migrations_matrix)

    cpx.variables.add(names=all_service_migrations_matrix,
                      lb=[0] * len(all_service_migrations_matrix),
                      ub=[1] * len(all_service_migrations_matrix),
                      types=["B"] * len(all_service_migrations_matrix))

    # 10) user_application_id[u][a][id] -> binary
    for u in range(sim.NUM_USER_REQUESTS):
        user_application_id.append([])
        for a in range(sim.NUM_APPLICATIONS):
            user_application_id[u].append([])
            for id in range(sim.MAX_IDS):
                varname = "user_application_id_" + str(u) + "_" + str(a) + "_" + str(id)
                all_user_application_id.append(varname)
                user_application_id[u][a].append(varname)

    cpx.variables.add(names=all_user_application_id,
                      lb=[0] * len(all_user_application_id),
                      ub=[1] * len(all_user_application_id),
                      types=["B"] * len(all_user_application_id))

    # 11) user_service_association[u][a][id][s][rep][n] -> binary
    for u in range(sim.NUM_USER_REQUESTS):
        user_service_association.append([])
        for a in range(sim.NUM_APPLICATIONS):
            user_service_association[u].append([])
            for id in range(sim.MAX_IDS):
                user_service_association[u][a].append([])
                for s in range(sim.NUM_SERVICES):
                    user_service_association[u][a][id].append([])
                    for rep in range(sim.MAX_REPLICAS):
                        user_service_association[u][a][id][s].append([])
                        for n in range(sim.NUM_NODES):
                            varname = "user_service_association_" + str(u) + "_" + str(a) + "_" + str(id) + "_" + str(
                                s) + "_" + str(rep) + "_" + str(n)
                            all_user_service_association.append(varname)
                            user_service_association[u][a][id][s][rep].append(varname)

    # print(all_user_service_association)

    cpx.variables.add(names=all_user_service_association,
                      lb=[0] * len(all_user_service_association),
                      ub=[1] * len(all_user_service_association),
                      types=["B"] * len(all_user_service_association))

    # 12) user_delay[u] -> binary
    for u in range(sim.NUM_USER_REQUESTS):
        varname = "user_delay_" + str(u)
        user_delay.append(varname)

    #print(user_delay)

    cpx.variables.add(names=user_delay,
                      lb=[0] * len(user_delay),
                      ub=[cplex.infinity] * len(user_delay),
                      types=["N"] * len(user_delay))


def basic_constraints(sim, cpx):
    # Constraints Migrations
    add_constraint_previous_placement_matrix(sim, cpx, previous_placement_matrix, sim.previous_placement_matrix)
    add_constraint_migrations(sim, cpx, migrations_matrix, previous_placement_matrix, placement_matrix)
    add_constraint_migrations_2nd(sim, cpx, migrations_matrix, service_migrations_matrix)

    #if sim.run_number != 1:
        #print("Dynamic Case: Restrict migrations!")
        #add_constraint_restrict_migrations(sim, cpx, replicas_per_service, service_migrations_matrix)

    # Constraints User per SFC
    add_constraint_sfc_slots_limit(sim, cpx, user_application_id)
    add_constraint_users_per_sfc(sim, cpx, user_application_id)

    # Constraints User per Services
    add_constraint_service_slots_limit(sim, cpx, user_service_association)
    add_constraint_association_total_services(sim, cpx, user_service_association)
    add_constraint_association_one_per_service(sim, cpx, user_service_association)
    add_constraint_association_service_placement(sim, cpx, user_service_association, placement_matrix)

    # Cloud: Constraints
    # CPU / MEM / Band limit
    add_constraint_cpu_mem_band_limits(sim, cpx, placement_matrix)

    # Constraint:
    # Services executed on the network
    add_constraint_services_executed_on_the_network(sim, cpx, placement_matrix, acceptance_matrix)
    add_constraint_acceptance_request(sim, cpx, acceptance_matrix, acceptance_service_matrix, placement_matrix,
                                      replicas_per_service)

    # Constraint: 1 service request per 1 node
    add_constraint_one_service_request_per_node(sim, cpx, placement_matrix)

    # Constraint: Ensuring Us,n can only take value 0 if service is not used for any App request
    add_constraint_execution_matrix(sim, cpx, placement_matrix, execution_matrix)

    # Flow conservation Constraints
    # Constraint.addConstraintFlowConservation(Sim, cplex, flowMatrix, placementMatrix, sfcDelay)

    # Extra constraints
    add_extra_constraint_objective_min_number_nodes(sim, cpx, execution_matrix, node_utilization_matrix)

    # Contraint same service on different nodes
    add_constraint_same_service_different_nodes(sim, cpx, placement_matrix)
    add_constraint_service_based_instance_matrix(sim, cpx, placement_matrix)

    # Constraint User Delay
    add_constraint_user_delay(sim, cpx, user_delay, user_service_association)


def objective(sim, objective, cpx):
    # Multiple Objectives
    if objective == "MAX_USER_REQUESTS":  # MAX REQUESTS
        #logging.info("Selected Objective: " + objective)
        cpx.objective.set_name("MAX_USER_REQUESTS")
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        cpx.objective.set_linear(objective_MAX_acceptance_user_requests(sim, user_application_id))

    elif objective == "MIN_REPLICAS":  # MIN_REPLICAS
        #logging.info("Selected Objective: " + objective)
        cpx.objective.set_name("MIN_REPLICAS")
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        cpx.objective.set_linear(objective_number_service_replicas(sim, replicas_per_service))

    elif objective == "MAX_REPLICAS":  # MAX_REPLICAS
        #logging.info("Selected Objective: " + objective)
        cpx.objective.set_name("MAX_REPLICAS")
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        cpx.objective.set_linear(objective_number_service_replicas(sim, replicas_per_service))

    elif objective == "MIN_NODES":  # MIN NODES
        #logging.info("Selected Objective: " + objective)
        cpx.objective.set_name("MIN_NODES")
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        cpx.objective.set_linear(objective_MIN_number_nodes(sim, node_utilization_matrix))

    elif objective == "MIN_COST":  # MIN COST
        logging.info("Selected Objective: " + objective)
        cpx.objective.set_name("MIN_COST")
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        cpx.objective.set_linear(objective_MIN_cost(sim, placement_matrix))

    elif objective == "MIN_USER_DELAY":  # MIN User Access Latency
        #logging.info("Selected Objective: " + objective)
        cpx.objective.set_name("MIN_USER_DELAY")
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        cpx.objective.set_linear(objective_MIN_user_access_delay(sim, user_delay))



def constraint_previous_objective(sim, cpx, prev_obj, value_obj):
    # Multiple Objectives
    if prev_obj == "MAX_USER_REQUESTS":  # MAX REQUESTS
        add_constraint_objective_max_acceptance_user_requests(sim, cpx, user_application_id, value_obj)
    elif prev_obj == "MIN_REPLICAS":  # MIN_REPLICAS
        add_constraint_objective_number_service_replicas(sim, cpx, replicas_per_service, value_obj)
    elif prev_obj == "MAX_REPLICAS":  # MAX_REPLICAS
        add_constraint_objective_number_service_replicas(sim, cpx, replicas_per_service, value_obj)
    elif prev_obj == "MIN_NODES":  # NODES
        add_constraint_objective_min_nodes(sim, cpx, node_utilization_matrix, value_obj)
    if prev_obj == "MIN_COST":  # MIN COST
        add_constraint_objective_min_cost(sim, cpx, placement_matrix, value_obj)
    elif prev_obj == "MIN_USER_DELAY":  # MIN User Delay
        add_constraint_objective_min_user_delay(sim, cpx, user_delay, value_obj)


def first_iteration(sim, cpx, op1):
    #logging.info("\n----------------------Start Iteration--------------------------- ")

    # Reset Variables from previous iterations
    reset_variables()

    # variables
    initializa_variables(sim, cpx)

    # contraints
    basic_constraints(sim, cpx)

    # objective
    objective(sim, op1, cpx)

    # Solve model
    # Branch and Bound options
    # MIP Options
    cpx.parameters.mip.strategy.nodeselect.set(
        1)  # 0 - Depth-first search / 1 - Best-bound search (default) / 2 - Best-estimate search
    cpx.parameters.mip.strategy.branch.set(1)  # 0 - Automatic / 1- Branch UP - change to automatic?
    cpx.parameters.timelimit.set(3600)  # Time limit of 12h -> 43 200 s / 4h -> 14400 / 2h ->  7200s / 30 min: 1800s / 1h: 3600s

    # if sim.NUM_USER_REQUESTS > 40:
    #    cpx.parameters.mip.tolerances.mipgap.set(0.45) #0.15
    #    print("Gap tolerance = ", cpx.parameters.mip.tolerances.mipgap.get())

    # Display options
    cpx.parameters.mip.display.set(2) #2
    cpx.parameters.tune.display.set(2) #2
    cpx.parameters.simplex.display.set(2) #2

    # print(cpx.objective.sense[cpx.objective.get_sense()])
    # print(cpx.objective.get_linear())
    # print(cpx.objective.get_name())

    sim.time_start = cpx.get_time()

    # print("Variables: " + str(cpx.variables.get_num()))
    # print("Names:" + str(cpx.variables.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Indicator constraints: " + str(cpx.indicator_constraints.get_num()))
    # print("Names:" + str(cpx.indicator_constraints.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Linear Constraints: " + str(cpx.linear_constraints.get_num()))
    # print("Names:" + str(cpx.linear_constraints.get_names()))
    # print("-------------------------------------------------------------------")

    cpx.solve()
    finish(sim, cpx)

    return cplex


def second_iteration(sim, cpx, op2):
    #logging.info("\n----------------------Start Iteration--------------------------- ")
    # Reset Variables
    reset_variables()

    # variables
    initializa_variables(sim, cpx)

    # contraints
    basic_constraints(sim, cpx)

    # Constraint of First Iteration added
    constraint_previous_objective(sim, cpx, sim.op1, sim.obj1)

    # objective
    objective(sim, op2, cpx)

    # Solve model
    # Branch and Bound options
    # MIP Options
    cpx.parameters.mip.strategy.nodeselect.set(
        1)  # 0 - Depth-first search / 1 - Best-bound search (default) / 2 - Best-estimate search
    cpx.parameters.mip.strategy.branch.set(1)  # 0 - Automatic / 1- Branch UP - change to automatic?
    cpx.parameters.timelimit.set(100000)  # Time limit of 12h -> 43 200 s / 4h -> 14400 / 2h ->  7200s / 30 min: 1800s

    # Display options
    cpx.parameters.mip.display.set(2)
    cpx.parameters.tune.display.set(2)
    cpx.parameters.simplex.display.set(2)

    # print(cpx.objective.sense[cpx.objective.get_sense()])
    # print(cpx.objective.get_linear())
    # print(cpx.objective.get_name())

    sim.time_start = cpx.get_time()

    # print("Variables: " + str(cpx.variables.get_num()))
    # print("Names:" + str(cpx.variables.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Indicator constraints: " + str(cpx.indicator_constraints.get_num()))
    # print("Names:" + str(cpx.indicator_constraints.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Linear Constraints: " + str(cpx.linear_constraints.get_num()))
    # print("Names:" + str(cpx.linear_constraints.get_names()))
    # print("-------------------------------------------------------------------")

    cpx.solve()
    finish(sim, cpx)

    return cplex


def third_iteration(sim, cpx, op3):
    #logging.info("\n----------------------Start Iteration--------------------------- ")
    # Reset Variables
    reset_variables()

    # variables
    initializa_variables(sim, cpx)

    # contraints
    basic_constraints(sim, cpx)

    # Constraint of First Iteration added
    constraint_previous_objective(sim, cpx, sim.op1, sim.obj1)

    # Constraint of Second Iteration added
    constraint_previous_objective(sim, cpx, sim.op2, sim.obj2)

    # objective
    objective(sim, op3, cpx)

    # Solve model
    # Branch and Bound options
    # MIP Options
    cpx.parameters.mip.strategy.nodeselect.set(
        1)  # 0 - Depth-first search / 1 - Best-bound search (default) / 2 - Best-estimate search
    cpx.parameters.mip.strategy.branch.set(1)  # 0 - Automatic / 1- Branch UP - change to automatic?
    cpx.parameters.timelimit.set(100000)  # Time limit of 12h -> 43 200 s / 4h -> 14400 / 2h ->  7200s / 30 min: 1800s

    # Display options
    cpx.parameters.mip.display.set(2)
    cpx.parameters.tune.display.set(2)
    cpx.parameters.simplex.display.set(2)

    # print(cpx.objective.sense[cpx.objective.get_sense()])
    # print(cpx.objective.get_linear())
    # print(cpx.objective.get_name())

    sim.time_start = cpx.get_time()

    # print("Variables: " + str(cpx.variables.get_num()))
    # print("Names:" + str(cpx.variables.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Indicator constraints: " + str(cpx.indicator_constraints.get_num()))
    # print("Names:" + str(cpx.indicator_constraints.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Linear Constraints: " + str(cpx.linear_constraints.get_num()))
    # print("Names:" + str(cpx.linear_constraints.get_names()))
    # print("-------------------------------------------------------------------")

    cpx.solve()
    finish(sim, cpx)

    return cplex


def fourth_iteration(sim, cpx, op4):
    #logging.info("\n----------------------Start Iteration--------------------------- ")
    # Reset Variables
    reset_variables()

    # variables
    initializa_variables(sim, cpx)

    # contraints
    basic_constraints(sim, cpx)

    # Constraint of First Iteration added
    constraint_previous_objective(sim, cpx, sim.op1, sim.obj1)

    # Constraint of Second Iteration added
    constraint_previous_objective(sim, cpx, sim.op2, sim.obj2)

    # Constraint of Third Iteration added
    constraint_previous_objective(sim, cpx, sim.op3, sim.obj3)

    # objective
    objective(sim, op4, cpx)

    # Solve model
    # Branch and Bound options
    # MIP Options
    cpx.parameters.mip.strategy.nodeselect.set(
        1)  # 0 - Depth-first search / 1 - Best-bound search (default) / 2 - Best-estimate search
    cpx.parameters.mip.strategy.branch.set(1)  # 0 - Automatic / 1- Branch UP - change to automatic?
    cpx.parameters.timelimit.set(100000)  # Time limit of 12h -> 43 200 s / 4h -> 14400 / 2h ->  7200s / 30 min: 1800s

    # Display options
    cpx.parameters.mip.display.set(2)
    cpx.parameters.tune.display.set(2)
    cpx.parameters.simplex.display.set(2)

    # print(cpx.objective.sense[cpx.objective.get_sense()])
    # print(cpx.objective.get_linear())
    # print(cpx.objective.get_name())

    sim.time_start = cpx.get_time()

    # print("Variables: " + str(cpx.variables.get_num()))
    # print("Names:" + str(cpx.variables.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Indicator constraints: " + str(cpx.indicator_constraints.get_num()))
    # print("Names:" + str(cpx.indicator_constraints.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Linear Constraints: " + str(cpx.linear_constraints.get_num()))
    # print("Names:" + str(cpx.linear_constraints.get_names()))
    # print("-------------------------------------------------------------------")

    cpx.solve()
    finish(sim, cpx)

    return cplex


def fifth_iteration(sim, cpx, op5):
    #logging.info("\n----------------------Start Iteration--------------------------- ")
    # Reset Variables
    reset_variables()

    # variables
    initializa_variables(sim, cpx)

    # contraints
    basic_constraints(sim, cpx)

    # Constraint of First Iteration added
    constraint_previous_objective(sim, cpx, sim.op1, sim.obj1)

    # Constraint of Second Iteration added
    constraint_previous_objective(sim, cpx, sim.op2, sim.obj2)

    # Constraint of Third Iteration added
    constraint_previous_objective(sim, cpx, sim.op3, sim.obj3)

    # Constraint of Fourth Iteration added
    constraint_previous_objective(sim, cpx, sim.op4, sim.obj4)

    # objective
    objective(sim, op5, cpx)

    # Solve model
    # Branch and Bound options
    # MIP Options
    cpx.parameters.mip.strategy.nodeselect.set(
        1)  # 0 - Depth-first search / 1 - Best-bound search (default) / 2 - Best-estimate search
    cpx.parameters.mip.strategy.branch.set(1)  # 0 - Automatic / 1- Branch UP - change to automatic?
    cpx.parameters.timelimit.set(100000)  # Time limit of 12h -> 43 200 s / 4h -> 14400 / 2h ->  7200s / 30 min: 1800s

    # Display options
    cpx.parameters.mip.display.set(2)
    cpx.parameters.tune.display.set(2)
    cpx.parameters.simplex.display.set(2)

    # print(cpx.objective.sense[cpx.objective.get_sense()])
    # print(cpx.objective.get_linear())
    # print(cpx.objective.get_name())

    sim.time_start = cpx.get_time()

    # print("Variables: " + str(cpx.variables.get_num()))
    # print("Names:" + str(cpx.variables.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Indicator constraints: " + str(cpx.indicator_constraints.get_num()))
    # print("Names:" + str(cpx.indicator_constraints.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Linear Constraints: " + str(cpx.linear_constraints.get_num()))
    # print("Names:" + str(cpx.linear_constraints.get_names()))
    # print("-------------------------------------------------------------------")

    cpx.solve()
    finish(sim, cpx)

    return cplex


def sixth_iteration(sim, cpx, op6):
    #logging.info("\n----------------------Start Iteration--------------------------- ")
    # Reset Variables
    reset_variables()

    # variables
    initializa_variables(sim, cpx)

    # contraints
    basic_constraints(sim, cpx)

    # Constraint of First Iteration added
    constraint_previous_objective(sim, cpx, sim.op1, sim.obj1)

    # Constraint of Second Iteration added
    constraint_previous_objective(sim, cpx, sim.op2, sim.obj2)

    # Constraint of Third Iteration added
    constraint_previous_objective(sim, cpx, sim.op3, sim.obj3)

    # Constraint of Fourth Iteration added
    constraint_previous_objective(sim, cpx, sim.op4, sim.obj4)

    # Constraint of Fifth Iteration added
    constraint_previous_objective(sim, cpx, sim.op5, sim.obj5)

    # objective
    objective(sim, op6, cpx)

    # Solve model
    # Branch and Bound options
    # MIP Options
    cpx.parameters.mip.strategy.nodeselect.set(
        1)  # 0 - Depth-first search / 1 - Best-bound search (default) / 2 - Best-estimate search
    cpx.parameters.mip.strategy.branch.set(1)  # 0 - Automatic / 1- Branch UP - change to automatic?
    cpx.parameters.timelimit.set(100000)  # Time limit of 12h -> 43 200 s / 4h -> 14400 / 2h ->  7200s / 30 min: 1800s

    # Display options
    cpx.parameters.mip.display.set(2)
    cpx.parameters.tune.display.set(2)
    cpx.parameters.simplex.display.set(2)

    # print(cpx.objective.sense[cpx.objective.get_sense()])
    # print(cpx.objective.get_linear())
    # print(cpx.objective.get_name())

    sim.time_start = cpx.get_time()

    # print("Variables: " + str(cpx.variables.get_num()))
    # print("Names:" + str(cpx.variables.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Indicator constraints: " + str(cpx.indicator_constraints.get_num()))
    # print("Names:" + str(cpx.indicator_constraints.get_names()))
    # print("-------------------------------------------------------------------")
    # print("Linear Constraints: " + str(cpx.linear_constraints.get_num()))
    # print("Names:" + str(cpx.linear_constraints.get_names()))
    # print("-------------------------------------------------------------------")

    cpx.solve()
    finish(sim, cpx)

    return cplex


def finish(sim, cpx):
    sim.time_end = cpx.get_time()
    sim.number_iterations = sim.number_iterations + 1

    sol = cpx.solution

    # the following line prints the corresponding string
    #logging.info(sol.status[sol.get_status()])

    #if sol.is_primal_feasible():
        #print("Solution value  = ", sol.get_objective_value())

    nodes = np.zeros(sim.NUM_NODES)

    # Print User Application ID
    '''
    print("\n---------------User request Acceptance------------------- ")
    for u in range(sim.NUM_USER_REQUESTS):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                if sol.get_values(user_application_id[u][a][id]) == 1.0:
                    print("User App ID [u" + str(u + 1) + "][a" + str(a + 1) + "][id" + str(
                        id + 1) + "] = " + str(sol.get_values(user_application_id[u][a][id])))

    # App Acceptance
    print("\n---------------App Acceptance------------------- ")
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            if sol.get_values(acceptance_matrix[a][id]) == 1.0:
                print(
                    "G [a" + str(a + 1) + "][id" + str(id + 1) + "] = " + str(sol.get_values(acceptance_matrix[a][id])))

    # Service Acceptance
    print("\n---------------Service Acceptance------------------- ")
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    if sol.get_values(acceptance_service_matrix[a][id][s]) == 1.0:
                        print("G [a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "] = " + str(
                            sol.get_values(acceptance_service_matrix[a][id][s])))
    '''
    replicas = 0
    copy_replicas_per_service = []
    #print("\n---------------Replicas per Service------------------- ")
    for a in range(sim.NUM_APPLICATIONS):
        copy_replicas_per_service.append([])
        for id in range(sim.MAX_IDS):
            copy_replicas_per_service[a].append([])
            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    replicas = replicas + sol.get_values(replicas_per_service[a][id][s])
                    copy_replicas_per_service[a][id].append(sol.get_values(replicas_per_service[a][id][s]))
                    #print("Service Replicas [a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "] = " + str(sol.get_values(replicas_per_service[a][id][s])))

    #print("Total Number of Service Replicas = " + str(replicas))

    sim.number_replicas = replicas
    sim.replicas_per_service = copy_replicas_per_service

    # Update for next iteration!
    copy_user_service_association = []
    # User App ID Service Replica Node
    #print("\n---------------User App ID Service Replica Node------------------- ")
    for u in range(sim.NUM_USER_REQUESTS):
        copy_user_service_association.append([])
        for a in range(sim.NUM_APPLICATIONS):
            copy_user_service_association[u].append([])
            for id in range(sim.MAX_IDS):
                copy_user_service_association[u][a].append([])
                for s in range(sim.NUM_SERVICES):
                    copy_user_service_association[u][a][id].append([])
                    for rep in range(sim.MAX_REPLICAS):
                        copy_user_service_association[u][a][id][s].append([])
                        for n in range(sim.NUM_NODES):
                            copy_user_service_association[u][a][id][s][rep].append(sol.get_values(user_service_association[u][a][id][s][rep][n]))
                            #if sol.get_values(user_service_association[u][a][id][s][rep][n]) == 1.0:
                                #print("User App ID Service Replica Node [u" + str(u + 1) + "][a" + str(
                                    #a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep" + str(
                                    #rep + 1) + "][n" + str(n + 1) + "] = " + str(sol.get_values(
                                    #user_service_association[u][a][id][s][rep][n])))

    #print("\n-------------------Service Slots Calculated Limits ------------------\n")
    '''
    service_slots = np.zeros((sim.NUM_APPLICATIONS, sim.MAX_IDS, sim.NUM_SERVICES, sim.MAX_REPLICAS, sim.NUM_NODES))

    for u in range(sim.NUM_USER_REQUESTS):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                for s in range(sim.NUM_SERVICES):
                    if (sim.instance[a][s] == 1):
                        for rep in range(sim.MAX_REPLICAS):
                            for n in range(sim.NUM_NODES):
                                if sol.get_values(user_service_association[u][a][id][s][rep][n]) == 1.0:
                                    # print("[a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep " + str(rep + 1)+ "][n" + str(n + 1) + "] -  Value:" + str(service_slots[a][id][s][rep][n]))
                                    service_slots[a][id][s][rep][n] = float(sim.user_cost[u]) + float(
                                        service_slots[a][id][s][rep][n])
                                    # print("[a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep " + str(rep + 1) + "][n" + str(n + 1) + "] -  New Value:" + str(service_slots[a][id][s][rep][n]))

    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                for rep in range(sim.MAX_REPLICAS):
                    for n in range(sim.NUM_NODES):
                        if service_slots[a][id][s][rep][n] != 0:
                            # print("Value: " + str(service_slots[a][id][s][rep][n]))
                            print("[a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep " + str(
                                rep + 1) + "][n" + str(n + 1) + "]: " + "Slots Allocated/Capacity:  " + str(
                                service_slots[a][id][s][rep][n]) + "/" + str(sim.service_slots[s]))
    '''
    #print("\n----------User Access Delay (E2E Delay = 2 x Value!)---------")

    delay = []
    e2e = []
    copy_user_delay = []

    for u in range(sim.NUM_USER_REQUESTS):
        #print("User Access Latency [u" + str(u + 1) + "] = " + str(sol.get_values(user_delay[u])) + " ms")
        copy_user_delay.append(sol.get_values(user_delay[u]))
        delay.append(sol.get_values(user_delay[u]))
        e2e.append(2 * delay[u])

    #print("Total User Delay: " + str(sum(delay)) + " ms")
    #print("MAX User Delay:  " + str(max(delay)) + " ms")
    #print("MIN User Delay:  " + str(min(delay)) + " ms")
    #print("AVG User Delay:  " + str(average(delay)) + " ms")
    #if len(delay) > 1:
        #print("Standard Deviation:  " + str(statistics.stdev(delay)) + " ms")

    '''
    print("\n---------------E2E Delay------------------- ")
    for u in range(sim.NUM_USER_REQUESTS):
        print("E2E Delay [u" + str(u + 1) + "] = " + str(e2e[u]) + " ms")

    print("Total E2E Delay: " + str(sum(e2e)) + " ms")
    print("MAX User E2E Delay:  " + str(max(e2e)) + " ms")
    print("MIN User E2E Delay:  " + str(min(e2e)) + " ms")
    print("AVG User E2E Delay:  " + str(average(e2e)) + " ms")
    if len(delay) > 1:
        print("Standard Deviation:  " + str(statistics.stdev(e2e)) + " ms")

    # Previous Placement Matrix
    print("\n---------------Previous Placement Matrix------------------- ")
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                for rep in range(sim.MAX_REPLICAS):
                    for n in range(sim.NUM_NODES):
                        if sol.get_values(previous_placement_matrix[a][id][s][rep][n]) != 0.0:
                            print("P[a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep " + str(
                                rep + 1) + "][n" + str(n + 1) + "] = " + str(
                                sol.get_values(previous_placement_matrix[a][id][s][rep][n])))
    '''
    # Placement Matrix
    copy_placement_matrix = []
    #print("\n---------------Placement Matrix------------------- ")
    for a in range(sim.NUM_APPLICATIONS):
        copy_placement_matrix.append([])
        for id in range(sim.MAX_IDS):
            copy_placement_matrix[a].append([])
            for s in range(sim.NUM_SERVICES):
                copy_placement_matrix[a][id].append([])
                for rep in range(sim.MAX_REPLICAS):
                    copy_placement_matrix[a][id][s].append([])
                    for n in range(sim.NUM_NODES):
                        copy_placement_matrix[a][id][s][rep].append(sol.get_values(placement_matrix[a][id][s][rep][n]))
                        if sol.get_values(placement_matrix[a][id][s][rep][n]) == 1.0:
                            nodes[n] = 1
                            #print("P[a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep " + str(
                                #rep + 1) + "][n" + str(n + 1) + "] = " + str(
                                #sol.get_values(placement_matrix[a][id][s][rep][n])))
    '''
    # Migration Matrix
    print("\n--------------- Migrations Matrix------------------- ")
    count_migrations = 0
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    for rep in range(sim.MAX_REPLICAS):
                        for n in range(sim.NUM_NODES):
                            if sol.get_values(migrations_matrix[a][id][s][rep][n]) == 1.0:
                                count_migrations = count_migrations + 1
                                print("M[a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep " + str(
                                    rep + 1) + "][n " + str(n + 1) + "] = " + str(
                                    sol.get_values(migrations_matrix[a][id][s][rep][n])))

    # Service Migration Matrix
    count_service_migrations = 0
    print("\n--------------- Service Migration Matrix------------------- ")
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    for rep in range(sim.MAX_REPLICAS):
                        if sol.get_values(service_migrations_matrix[a][id][s][rep]) == 1.0:
                            count_service_migrations = count_service_migrations + 1
                            print("SM[a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep " + str(
                                rep + 1) + "] = " + str(sol.get_values(service_migrations_matrix[a][id][s][rep])))

    print("Number of Migrations (M):  " + str(count_migrations))
    print("Number of Service Migrations (SM):  " + str(count_service_migrations))
    print("Total Number of Replicas:  " + str(sim.number_replicas))
    print("Percentage of Migrations:  " + str((count_service_migrations / sim.number_replicas) * 100) + " (%)")
    '''

    # Update for Comparison
    sim.milp_solution_previous_placement_matrix = sim.previous_placement_matrix
    sim.milp_solution_placement_matrix = copy_placement_matrix
    sim.milp_solution_user_service_association = copy_user_service_association
    sim.milp_solution_user_delay = copy_user_delay

    # Update for next iteration!
    sim.previous_placement_matrix = copy_placement_matrix

    '''
    # Node Utilization
    count = 0
    print("\n---------------Node usage------------------- ")
    for n in range(sim.NUM_NODES):
        if sol.get_values(node_utilization_matrix[n]) == 1.0:
            count = count + 1

    print("Active Nodes (%):  " + str((count / sim.NUM_NODES) * 100))

    # Real Node Utilization
    count = 0
    print("\n---------------Real Node usage------------------- ")
    for n in range(sim.NUM_NODES):
        if nodes[n] == 1:
            count = count + 1
            print("Node [n" + str(n + 1) + "] = " + str(nodes[n]))

    print("Active Nodes (%):  " + str((count / sim.NUM_NODES) * 100))

    print("\n-------------Node Calculated Limits --------------------------\n")
    # CPU MEM BAND
    limits_mem = np.zeros(sim.NUM_NODES)
    limits_cpu = np.zeros(sim.NUM_NODES)
    limits_band = np.zeros(sim.NUM_NODES)
    for n in range(sim.NUM_NODES):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                for s in range(sim.NUM_SERVICES):
                    for rep in range(sim.MAX_REPLICAS):
                        if sol.get_values(placement_matrix[a][id][s][rep][n]) == 1.0:
                            limits_mem[n] = float(sim.service_mem[s]) + float(limits_mem[n])
                            limits_cpu[n] = float(sim.service_cpu[s]) + float(limits_cpu[n])
                            limits_band[n] = float(sim.service_band[s]) + float(limits_band[n])

    for n in range(sim.NUM_NODES):
        print("n" + str(n + 1) + " CPU Allocated/Capacity:  " + str(limits_cpu[n]) + "/" + str(sim.node_cpu[n]))
        print("n" + str(n + 1) + " MEM Allocated/Capacity:  " + str(limits_mem[n]) + "/" + str(sim.node_mem[n]))
        print("n" + str(n + 1) + " BAND Allocated/Capacity:  " + str(limits_band[n]) + "/" + str(sim.node_band[n]))
        print("\n------------------------------------------------- ")
    '''
    #logging.info("\n----------------------End Iteration--------------------------- ")

