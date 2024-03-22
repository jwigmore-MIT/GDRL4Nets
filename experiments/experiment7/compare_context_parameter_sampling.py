import json
from generate_context_set import sample_contexts_dilkins, sample_contexts_hit_and_run
from context_set_stats import plot_arrival_rate_histogram

num_contexts = 10
thin = 1000
sampling_method = 'dilkins' #or 'hit_and_run'
context_space_dict = json.load(open("SH2u_lf1.32_context_space-nondominated.json", 'rb'))
hr_context_samples = sample_contexts_hit_and_run(context_space_dict, num_contexts, thin = thin)

dilkins_context_samples = sample_contexts_dilkins(context_space_dict, num_contexts)
# plot the context samples
plot_arrival_rate_histogram(hr_context_samples, title = "Hit and Run Sampling Parameters Histogram")
plot_arrival_rate_histogram(dilkins_context_samples, title = "Dikin Walk Sampling Parameters Histogram")
#plot_arrival_rate_histogram(dikin_context_samples)