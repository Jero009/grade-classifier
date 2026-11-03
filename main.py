import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_creation import create_data



def main():
    while True:
        try:
            data_rows = int(input("Vnesi stevilo linij podatkov: "))
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
            epoch_count = int(input("Vnesi stevilo epoh: "))
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

    create_data(data_rows) # generira dataset 


    # 1. naloži podatke iz CSV datoteke in jih pretvori v PyTorch tenzorje
    data = np.loadtxt("data.csv", delimiter=",", skiprows=1) #pythorch tenzor
    X = torch.tensor(data[:, 0:1], dtype=torch.float32) / 100.0  # procente spremenimo med 0 in 1
    y = torch.tensor(data[:, 1] - 1, dtype=torch.long)          # ocene premaknmo da so od 0 do 4 

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
    model = nn.Linear(1, 5)                      # Linearni klasifikator, ki sprejme 1 vhodno funkcijo (procent) in vrne 5 izhodnih vrednosti (grade 0-4)
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
        print(f"\nKončni Accuracy modela: {accuracy.item() * 100:.1f}%")

if __name__ == "__main__":
    main()