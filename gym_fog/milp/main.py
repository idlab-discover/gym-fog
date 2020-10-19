from gym_fog.milp.simulation import Simulation
from gym_fog.milp import ilp

import numpy as np

if __name__ == '__main__':
    Users = 48
    runNumber = 1
    sim = Simulation("CPLEX_JOURNAL_JNCA", Users, runNumber, "MIN_COST", "NONE", "NONE", "NONE", "NONE","NONE")

    #print(sim.name)
    ilp.run(sim)

    #new_request = sim.NUM_USER_REQUESTS + 1
    #sim.increase_number_users(new_request)
    #sim.run_number = sim.run_number + 1
    # sim.change_user_locations()
    #ilp.run(sim)

