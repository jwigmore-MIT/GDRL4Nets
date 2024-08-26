from argparse import ArgumentParser
import os
from datetime import datetime
from generate_context_set import sample_contexts_hit_and_run, sample_contexts_dilkins, create_context_set_dict
import json
from torchrl_development.utils.configuration import make_serializable

if __name__ == "__main__":
    parser = ArgumentParser(description='Run experiment')

    parser.add_argument('--context_space_json', type=str, help='path to the context space json file', default="SH4_lf1.5_context_space-nondominated.json")
    parser.add_argument('--num_contexts', type=int, help='number of contexts to sample', default=50)
    parser.add_argument('--sampling_method', type=str, help='sampling method to use', default='hit_and_run')
    parser.add_argument('--max_rollout_steps', type=int, help='maximum number of rollout steps', default=50_000)
    parser.add_argument('--num_rollouts_per_env', type=int, help='number of rollouts per environment', default=3)

    args = parser.parse_args()
    base_name = args.context_space_json.split("_")[0]

    context_space_dict = json.load(open(args.context_space_json, 'rb'))


    num_contexts = args.num_contexts
    thin = 1000
    sampling_method = args.sampling_method # 'dilkins' or 'hit_and_run'
    date_time = datetime.now().strftime('%m%d%H%M')

    # create folder to store all the sampled context parameters and context set dictionary
    folder_name = f"{base_name}_sampled_contexts_{num_contexts}_{sampling_method}_{date_time}"
    # if folder doesn't exist, create it
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    param_save_file = f"{base_name}_sampled_context_parameters_{num_contexts}_{sampling_method}_{date_time}.json"
    context_set_dict_file_name = f"{base_name}_context_set_{num_contexts}_{date_time}.json"
    if sampling_method == 'hit_and_run':
        print("Starting Hit and Run Sampling")
        context_samples = sample_contexts_hit_and_run(context_space_dict, num_contexts, thin=thin)
    else:
        context_samples = sample_contexts_dilkins(context_space_dict, num_contexts)

    make_env_params = {"observe_lambda": False,
                       "seed": None,
                       "terminal_backlog": None,
                       "observation_keys": ["Q", "Y"],
                       "inverse_reward": False,
                       "stat_window_size": 100000,
                       "terminate_on_convergence": False,
                       "convergence_threshold": 0.01}

    context_set_dict = create_context_set_dict(context_samples,
                                               context_space_dict["context_dicts"]["0"]["env_params"],
                                               make_env_params,
                                               max_rollout_steps= args.max_rollout_steps,
                                               num_rollouts_per_env=args.num_rollouts_per_env,
                                               plot = False)
    serial_context_set_dictionary = make_serializable(context_set_dict)
    with open(os.path.join(folder_name, context_set_dict_file_name), "w") as f:
        json.dump(serial_context_set_dictionary, f)