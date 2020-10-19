import statistics

import cplex
from gym_fog.milp.model import first_iteration, second_iteration, third_iteration, fourth_iteration, fifth_iteration, \
    sixth_iteration
from gym_fog.milp.util import average


def run(sim):
    # First Iteration
    cpx1 = cplex.Cplex()

    first_iteration(sim, cpx1, sim.op1)

    sol = cpx1.solution
    sim.obj1 = sol.get_objective_value()

    sim.duration_IT1 = float("{:.3f}".format(sim.time_end - sim.time_start))
    sim.duration = float("{:.3f}".format(sim.duration + sim.duration_IT1))

    cpx1.end()

    if sim.op2 == "NONE":
        finish(sim)
    else:
        # Second Iteration
        cpx2 = cplex.Cplex()

        second_iteration(sim, cpx2, sim.op2)

        sol = cpx2.solution
        sim.obj2 = sol.get_objective_value()

        sim.duration_IT2 = float("{:.3f}".format(sim.time_end - sim.time_start))
        sim.duration = float("{:.3f}".format(sim.duration + sim.duration_IT2))

        cpx2.end()

        if sim.op3 == "NONE":
            finish(sim)
        else:
            # Third Iteration
            cpx3 = cplex.Cplex()

            third_iteration(sim, cpx3, sim.op3)

            sol = cpx3.solution
            sim.obj3 = sol.get_objective_value()

            sim.duration_IT3 = float("{:.3f}".format(sim.time_end - sim.time_start))
            sim.duration = float("{:.3f}".format(sim.duration + sim.duration_IT3))

            cpx3.end()

            if sim.op4 == "NONE":
                finish(sim)
            else:
                # Fourth Iteration
                cpx4 = cplex.Cplex()

                fourth_iteration(sim, cpx4, sim.op4)

                sol = cpx4.solution
                sim.obj4 = sol.get_objective_value()

                sim.duration_IT4 = float("{:.3f}".format(sim.time_end - sim.time_start))
                sim.duration = float("{:.3f}".format(sim.duration + sim.duration_IT4))

                cpx4.end()

                if sim.op5 == "NONE":
                    finish(sim)
                else:
                    # Fifth Iteration
                    cpx5 = cplex.Cplex()
                    fifth_iteration(sim, cpx5, sim.op5)

                    sol = cpx5.solution
                    sim.obj5 = sol.get_objective_value

                    sim.duration_IT5 = float("{:.3f}".format(sim.time_end - sim.time_start))
                    sim.duration = float("{:.3f}".format(sim.duration + sim.duration_IT5))

                    cpx5.end()

                    if sim.op6 == "NONE":
                        finish(sim)
                    else:
                        # Sixth Iteration
                        cpx6 = cplex.Cplex()
                        sixth_iteration(sim, cpx6, sim.op6)

                        sol = cpx6.solution
                        sim.obj6 = sol.get_objective_value()

                        sim.duration_IT6 = float("{:.3f}".format(sim.time_end - sim.time_start))
                        sim.duration = float("{:.3f}".format(sim.duration + sim.duration_IT6))

                        cpx6.end()
                        finish(sim)


def finish(sim):
    # Set duration of simulation to 0
    sim.total_execution = float("{:.3f}".format(sim.total_execution + sim.duration))
    sim.number_simulations = sim.number_simulations + 1
    sim.run_number = sim.run_number + 1

    # Print Results
    print_results(sim)


def print_results(sim):
    print("\n--------------------Summary------------------------------")
    '''
    print("---------------Initial Variables-------------------------")
    print("Total number of Runs: " + str(sim.number_runs))
    print("Run Number: " + str(sim.run_number))
    print("Total Number of App IDs: " + str(sim.MAX_IDS))
    print("Total Number of Apps: " + str(sim.NUM_APPLICATIONS))
    print("Total Number of Computational Nodes: " + str(sim.NUM_NODES))
    print("Total Number of Services: " + str(sim.NUM_SERVICES))
    print("Total Number of Users: " + str(sim.NUM_USER_REQUESTS))

    print("---------------Decision Variables---------------------")
    print("Total Number of Service Replicas: " + str(sim.number_replicas))
    '''
    print("\n---------------Replicas per Service------------------- ")
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    print("Service Replicas [a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "] = " +
                          str(sim.replicas_per_service[a][id][s]))

    print("\n---------------Objectives----------------------------\n")
    print("Op1: " + sim.op1)
    print("Op2: " + sim.op2)
    print("Op3: " + sim.op3)
    print("Op4: " + sim.op4)
    print("Op5: " + sim.op5)
    print("Op6: " + sim.op6)

    print("\n-------------Decision Variables Solution---------------\n")
    print("Obj1: " + str(sim.obj1))
    print("Obj2: " + str(sim.obj2))
    print("Obj3: " + str(sim.obj3))
    print("Obj4: " + str(sim.obj4))
    print("Obj5: " + str(sim.obj5))
    print("Obj6: " + str(sim.obj6))

    print("\n-------------Duration Values---------------------------\n")
    print("IT1: " + str(sim.duration_IT1) + " s")
    print("IT2: " + str(sim.duration_IT2) + " s")
    print("IT3: " + str(sim.duration_IT3) + " s")
    print("IT4: " + str(sim.duration_IT4) + " s")
    print("IT5: " + str(sim.duration_IT5) + " s")
    print("IT6: " + str(sim.duration_IT6) + " s")

    print("Simulation Duration: " + str(sim.duration) + " s")
    print("Total execution time: " + str(sim.total_execution) + " s")
    print("avg ExecutionTime: " + str((sim.total_execution / sim.number_runs)) + " s")

    '''
    print("\n-------------MILP Solutions---------------------------\n")
    print("-------------Previous Placement---------------------------\n")
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    for rep in range(sim.MAX_REPLICAS):
                        for n in range(sim.NUM_NODES):
                            if sim.milp_solution_previous_placement_matrix[a][id][s][rep][n] == 1.0:
                                print(
                                    "P^-1[a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep " + str(
                                        rep + 1) + "][n" + str(n + 1) + "] = " + str(
                                        sim.milp_solution_previous_placement_matrix[a][id][s][rep][n]))

    print("-------------Placement---------------------------\n")
    for a in range(sim.NUM_APPLICATIONS):
        for id in range(sim.MAX_IDS):
            for s in range(sim.NUM_SERVICES):
                if sim.instance[a][s] == 1:
                    for rep in range(sim.MAX_REPLICAS):
                        for n in range(sim.NUM_NODES):
                            if sim.milp_solution_placement_matrix[a][id][s][rep][n] == 1.0:
                                print("P[a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(s + 1) + "][rep " + str(
                                    rep + 1) + "][n" + str(n + 1) + "] = " + str(
                                    sim.milp_solution_placement_matrix[a][id][s][rep][n]))
    '''
    print("\n-------------User Service Association---------------------------\n")
    for u in range(sim.NUM_USER_REQUESTS):
        for a in range(sim.NUM_APPLICATIONS):
            for id in range(sim.MAX_IDS):
                for s in range(sim.NUM_SERVICES):
                    if sim.instance[a][s] == 1:
                        for rep in range(sim.MAX_REPLICAS):
                            for n in range(sim.NUM_NODES):
                                if sim.milp_solution_user_service_association[u][a][id][s][rep][n] == 1.0:
                                    print("U[u" + str(u + 1) + "][a" + str(a + 1) + "][id" + str(id + 1) + "][s" + str(
                                        s + 1) + "][rep " + str(
                                        rep + 1) + "][n" + str(n + 1) + "] = " + str(
                                        sim.milp_solution_user_service_association[u][a][id][s][rep][n]))

    print("\n-------------------------User Delay---------------------------\n")
    delay = []
    e2e = []
    for u in range(sim.NUM_USER_REQUESTS):
        print("User Access Latency [u" + str(u + 1) + "] = " + str(sim.milp_solution_user_delay[u]) + " ms")
        delay.append(sim.milp_solution_user_delay[u])
        e2e.append(2 * delay[u])

    print("Total User Delay: " + str(sum(delay)) + " ms")
    print("MAX User Delay:  " + str(max(delay)) + " ms")
    print("MIN User Delay:  " + str(min(delay)) + " ms")
    print("AVG User Delay:  " + str(average(delay)) + " ms")
    if len(delay) > 1:
        print("Standard Deviation:  " + str(statistics.stdev(delay)) + " ms")
