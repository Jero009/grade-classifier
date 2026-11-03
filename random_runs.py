from main import main
import random as r
import torch
st_modelov = int(input("Vnesi stevilo modelov za treniranje: "))

for i in range(st_modelov):
    data_rows = r.randint(1000, 1000000)
    epoch_count = r.randint(1, 10000)
    training_percent = r.randint(0, 80)
    eval_percent = r.randint(0, 15)
    print(f"Zagon {i+1}: data_rows={data_rows}, epoch_count={epoch_count}, training_percent={training_percent}, eval_percent={eval_percent}")
    device = torch.device("cuda")
    main(data_rows, epoch_count, training_percent, eval_percent,device,test = False) 