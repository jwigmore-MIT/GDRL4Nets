import torch
# load two model.pt and see if they are the same

model1 = torch.load('model_2905000.pt')
model2 = torch.load('model_2905000b.pt')

# check if the two models have identical weights
for key, value in model1.items():
    if torch.all(torch.eq(value, model2[key])):
        print(f"{key}: Same")
    else:
        print(f"{key}: Different")
