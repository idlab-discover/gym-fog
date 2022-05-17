import csv
import logging


def log_episode_results_q_learning(episode_number, number_of_episodes, episode_steps, episode_reward, explore_rate,
                                   learning_rate, execution_time, percentage_requests):
    logging.info("Episode {}/{}:".format((episode_number + 1), number_of_episodes))
    logging.info("Steps: " + str(episode_steps))
    logging.info("Reward: " + str(episode_reward))
    logging.info("(%) Accepted Requests: " + str(float("{:.2f}".format(percentage_requests))))
    logging.info("Explore Rate: " + str(explore_rate))
    logging.info("Learning Rate: " + str(learning_rate))
    logging.info("Execution Time: " + str(float("{:.2f}".format(execution_time))) + " s")
    # print("Episode " + str(episode_number) + " finished!")


def log_episode_results(episode_number, number_of_episodes, episode_steps,
                        episode_reward, execution_time, percentage_requests):
    logging.info("Episode {}/{}:".format((episode_number + 1), number_of_episodes))
    logging.info("Steps: " + str(episode_steps))
    logging.info("Reward: " + str(episode_reward))
    logging.info("(%) Accepted Requests: " + str(float("{:.2f}".format(percentage_requests))))
    logging.info("Execution Time: " + str(float("{:.2f}".format(execution_time))) + " s")
    # print("Episode " + str(episode_number) + " finished!")


def save_to_csv(file_name, number_of_episodes, cost_data, reward_data, time_data, request_data):
    file = open(file_name, 'w', newline='')
    with file:
        fields = ['Episode', 'AvgCost', 'Reward', 'ExecutionTime', 'AvgAcceptedRequests']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i in range(number_of_episodes + 1):
            writer.writerow(
                {'Episode': i,
                 'AvgCost': float("{:.2f}".format(cost_data[i])),
                 'Reward': float("{:.2f}".format(reward_data[i])),
                 'ExecutionTime': float("{:.2f}".format(time_data[i])),
                 'AvgAcceptedRequests': float("{:.2f}".format(request_data[i]))}
            )


def save_to_csv_latency(file_name, number_of_episodes, cost_data, reward_data, time_data, request_data, rl_latency, milp_latency, latency_diff):
    file = open(file_name, 'w', newline='')
    with file:
        fields = ['Episode', 'AvgCost', 'Reward', 'ExecutionTime', 'AvgAcceptedRequests', 'rlLat', 'milpLat', 'AvgLat']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i in range(number_of_episodes + 1):
            writer.writerow(
                {'Episode': i,
                 'AvgCost': float("{:.2f}".format(cost_data[i])),
                 'Reward': float("{:.2f}".format(reward_data[i])),
                 'ExecutionTime': float("{:.2f}".format(time_data[i])),
                 'AvgAcceptedRequests': float("{:.2f}".format(request_data[i])),
                 'rlLat': float("{:.2f}".format(rl_latency[i])),
                 'milpLat': float("{:.2f}".format(milp_latency[i])),
                 'AvgLat': float("{:.2f}".format(latency_diff[i]))
                 }
            )


def save_ilp_csv(max_requests, time_data):
    file = open('ilp.csv', 'w', newline='')
    with file:
        fields = ['Number of User Requests', 'Execution Time']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i in range(max_requests + 1):
            writer.writerow(
                {'Number of User Requests': i,
                 'Execution Time': float("{:.2f}".format(time_data[i]))}
            )