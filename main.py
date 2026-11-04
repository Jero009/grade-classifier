import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from datetime import datetime
from data_creation import create_data

def main_arg():
    # Vprašanje za izbiro GPU-ja
    while True:
        use_gpu = input("Ali želiš uporabiti GPU (če je na voljo)? (y/n): ").strip().lower()
        if use_gpu in ['y', 'n']:
            break
        print("Prosim vnesi 'y' za da ali 'n' za ne.")

    if use_gpu == 'y' and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Uporabljam napravo: GPU (CUDA)")
    else:
        device = torch.device("cpu")
        if use_gpu == 'y':
            print("GPU ni na voljo, zato uporabljam CPU.")
        else:
            print("Uporabljam napravo: CPU")

    while True:
        try:
            data_rows = int(input("Vnesi stevilo linij podatkov (max: 1,000,000): "))
            if data_rows <= 0:
                print("Prosim vnesi pozitivno število.")
                continue
            elif data_rows > 1000000:
                print("Prosim vnesi število manjše od 1,000,000.")
                continue
            break
        except ValueError:
            print("Prosim vnesi int")
            
    while True:
        try:
            epoch_count = int(input("Vnesi stevilo epoh (max: 10,000): "))
            if epoch_count <= 0:
                print("Prosim vnesi pozitivno število.")
                continue
            elif epoch_count > 10000:
                print("Prosim vnesi število manjše od 10,000.")
                continue
            break
        except ValueError:
            print("Prosim vnesi int")

    while True:
        try:
            print("Prosim vnesi odstotke za treniranje in validacijo.")
            training_percent = float(input("vnsi odstotek podatkov za treniranje (0-80): "))
            eval_percent = float(input("vnsi odstotek podatkov za validacijo (0-15): "))
            if training_percent < 0 or training_percent > 80 or eval_percent < 0 or eval_percent > 15:
                print("vnesi odstotek med 0-80 za treniranje in 0-15 za validacijo")
                continue
            break
        except ValueError:
            print("Prosim vnesi int")

    # Vrnemo zbrane nastavitve, da jih lahko uporabimo v main funkciji
    return data_rows, epoch_count, training_percent, eval_percent, device


def main(data_rows, epoch_count, training_percent, eval_percent, device, test=True):
    create_data(data_rows) # generira dataset 

    # 1. naloži podatke iz CSV datoteke in jih pretvori v PyTorch tenzorje
    data = np.loadtxt("data.csv", delimiter=",", skiprows=1) #pythorch tenzor
    X = torch.tensor(data[:, 0:1], dtype=torch.float32).to(device) / 100.0  # procente spremenimo med 0 in 1
    y = torch.tensor(data[:, 1] - 1, dtype=torch.long).to(device)          # ocene premaknmo da so od 0 do 4 

    # 2. premešamo podatke (če bi uporabljali podatke, ki nebi bili naključno generirani, bi bilo to potrebno)
    num_samples = len(data)
    indices = torch.randperm(num_samples) # podatki

    # izračunamo velikost posameznih skupin glede na podani odstotek
    train_size = int(training_percent / 100 * num_samples)
    val_size = int(eval_percent / 100 * num_samples)

    # razdelmimo podatke na trening, validacijo in testiranje
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]  # preostanek podatkov bo za testiranje

    # ustvarimo posamezne tenzorje za trening, validacijo in testiranje

    #treniranje
    X_train = X[train_idx]
    y_train = y[train_idx]
    #validacija
    X_val = X[val_idx]
    y_val = y[val_idx]
    #testiranje
    X_test = X[test_idx]
    y_test = y[test_idx]

    # 3. ustvarimo model 
    #model = nn.Linear(1, 5).to(device)   
    #         # Linearni klasifikator, ki sprejme 1 vhodno funkcijo (procent) in vrne 5 izhodnih vrednosti (grade 0-4)
    model = nn.Sequential(
        nn.Linear(1,7), #vhodna plast ki sprejme 1 vhod
        nn.ReLU(), #aktivacijska funkcija ReLU
        nn.Linear(7,5) #izhodna plast ki vrne 5 izhodnih vrednosti (grade 0-4)
    ).to(device)
    criterion = nn.CrossEntropyLoss()            # Loss function ki meri razliko med napovedanimi in dejanskimi ocenami (grade 0-4)
    optimizer = optim.SGD(model.parameters(), lr=0.05) # Optimizer za posodabljanje teže modela na podlagi izračunanih gradientov

    # 4. treniranje modela na v epohah
    for epoch in range(epoch_count):
        optimizer.zero_grad()                    # Resetiramo gradientne vrednosti, da preprečimo akumulacijo iz prejšnjih epoh
        loss = criterion(model(X_train), y_train) # Forward pass: izračunamo napovedi modela in primerjamo z dejanskimi ocenami, da dobimo loss
        loss.backward()                          # Backward pass: backpropagation za izračun gradientov glede na loss
        optimizer.step()                         # posodobimo teže modela na podlagi izračunanih gradientov
        
        # Vsakih 200 epoh izpišemo validacijski loss, da spremljamo napredek modela
        if (epoch + 1) % 200 == 0:
            model.eval()                         # evaluacija modela
            with torch.no_grad():                
                val_loss = criterion(model(X_val), y_val) #izračunamo validacijski loss na validacijskem setu
                print(f"epoha {epoch+1} | Validacijski Loss: {val_loss.item():.4f}")
            model.train()                        # model se vrne v trenirni način, da lahko nadaljujemo s treniranjem

    # 5. Testiranje modela na testnem setu in izračun natančnosti (accuracy)
    model.eval()                                 # model nastavimo v eval mode
    with torch.no_grad():
        logits = model(X_test)                   # napovedi modela na testnem setu
        predictions = torch.argmax(logits, dim=1) # glede na izhodne vrednosti modela izberemo tisto z največjo verjetnostjo kot napovedano oceno
        accuracy = (predictions == y_test).float().mean() # izračunamo natančnost modela na testnem setu kot delež pravilno napovedanih ocen glede na dejanske ocene
        final_accuracy = accuracy.item() * 100
        print(f"\nKončni Accuracy modela: {final_accuracy:.1f}%")
        print(f"uporabljeni podatki: {data_rows} | treniranje: {train_size} | validacija: {val_size} | testiranje: {len(test_idx)} | število epoh: {epoch_count}")

    # SHRANJEVANJE ZAGONA (RUN-A) V MAPO "saved_models"
    models_dir = "saved_models"
    os.makedirs(models_dir, exist_ok=True)  # Ustvari mapo, če ne obstaja

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_epochs_{epoch_count}_rows_{data_rows}"
    model_filepath = os.path.join(models_dir, f"{run_name}.pth")
    
    # A) Shranimo uteži modela v mapo
    torch.save(model.state_dict(), model_filepath)
    
    # B) Pripravimo podatke za JSON zapis (shranimo relativno pot do modela)
    run_data = {
        "timestamp": timestamp,
        "data_rows": data_rows,
        "epoch_count": epoch_count,
        "training_percent": training_percent,
        "eval_percent": eval_percent,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": len(test_idx),
        "final_accuracy": round(final_accuracy, 2),
        "model_file": model_filepath
    }

    # C) Zapišemo/dodamo v skupno datoteko z vsemi zagoni
    history_file = "model_runs.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                runs_history = json.load(f)
            except json.JSONDecodeError:
                runs_history = []
    else:
        runs_history = []

    runs_history.append(run_data)

    with open(history_file, "w") as f:
        json.dump(runs_history, f, indent=4)

    print(f"Zagon uspešno shranjen. Model se nahaja v: {model_filepath}")

    # ročno preverjanje modela
    if test == True:
        while True:
            print("\n ročno previrjanje modela: ")
            try:
                percent = float(input("Vnesi procent (0-100) (vnesite 1234 za izhod): "))
                if percent < 0 or percent > 100:
                    print("Prosim vnesi procent med 0 in 100.")
                    continue
                if percent == 1234:
                    break
                
            except ValueError:
                print("Prosim vnesi float")
                continue
                
            model.eval() 
            with torch.no_grad():
                input_tensor = torch.tensor([[percent / 100.0]], dtype=torch.float32).to(device) # pretvorimo procent v tensor in normaliziramo med 0 in 1 ter prestavimo na napravo
                logits = model(input_tensor) # napoved modela (surove vrednosti)
                
                # Pretvori surove vrednosti (logits) v verjetnosti (odstotke) preko Softmax funkcije
                probabilities = torch.softmax(logits, dim=1)
                
                prediction = torch.argmax(logits, dim=1).item() + 1 # izberemo oceno z največjo verjetnostjo in jo prilagodimo na 1-5
                
                print(f"Napovedana ocena za {percent}% je {prediction}")
                print("Verjetnosti za posamezne ocene:")
                
                # Zanka za izpis verjetnosti za vsako oceno od 1 do 5
                for grade in range(1, 6):
                    prob = probabilities[0][grade - 1].item() * 100
                    print(f"Ocena {grade}: {prob:.1f}%")
                    for i in range(int(prob // 2)):
                        print("*", end="")
                    print() 

    
if __name__ == "__main__":
    # Pokličemo funkcijo za vnos argumentov in dobljene vrednosti posredujemo v main
    d_rows, e_count, t_perc, e_perc, dev = main_arg()
    main(d_rows, e_count, t_perc, e_perc, dev)