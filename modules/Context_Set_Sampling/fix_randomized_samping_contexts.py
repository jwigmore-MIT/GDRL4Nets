import json

context_set_dict = json.load(open("SH2u_context_set_l1_m3_s30.json", "r"))
contexts = {}
i =0
for context_key, context_dict in context_set_dict["contexts"].items():
    context_dict["lta"] = context_set_dict["ltas"][i]
    contexts[str(i)] = context_dict
    i+=1



new_context_set_dict = {"context_dicts": contexts, "ltas": context_set_dict["ltas"], "num_envs": context_set_dict["num_envs"]}

with open("n2SH2u_context_set_l1_m3_s30.json", "w") as f:
    json.dump(new_context_set_dict, f)