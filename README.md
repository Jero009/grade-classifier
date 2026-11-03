# Grade Classifier

Projekt uporablja PyTorch za napoved ocene od 1 do 5 na podlagi odstotka
uspešnosti. Podatki so trenutno umetno generirani, model pa je preprost
linearni klasifikator.

## Zahteve

- Python 3.10 ali novejši
- `pip`
- Za uporabo grafične kartice NVIDIA je potreben tudi ustrezen gonilnik CUDA.
  Program deluje tudi samo s procesorjem.

## Namestitev

1. Kloniraj repozitorij:

   ```bash
   git clone https://github.com/Jero009/grade-classifier.git
   cd grade-classifier
   ```

2. Priporočeno je ustvariti virtualno okolje:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   V Windows PowerShellu uporabi:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   V Windows Command Promptu uporabi:

   ```cmd
   .venv\Scripts\activate.bat
   ```

3. Namesti odvisnosti:

   ```bash
   python -m pip install -r requirements.txt
   ```

   Če želiš uporabljati CUDA, namesti različico PyTorcha, ki ustreza tvojemu
   sistemu, z uradne strani [PyTorch](https://pytorch.org/get-started/locally/).

Na Windowsu lahko ukaz `python` po potrebi zamenjaš z `py`, na Linuxu in
macOS pa s `python3`, če je tako nastavljeno v tvojem sistemu.

## Uporaba

### Treniranje enega modela

Zaženi:

```bash
python main.py
```

Program te bo vprašal:

- ali želiš uporabiti GPU (`y` ali `n`),
- število vrstic umetno ustvarjenih podatkov (največ 1.000.000),
- število učnih epoh (največ 10.000),
- odstotek podatkov za treniranje (0–80),
- odstotek podatkov za validacijo (0–15).

Preostali podatki se uporabijo za testiranje. Po koncu lahko vneseš odstotek
in program napove oceno ter izpiše verjetnosti za ocene od 1 do 5. Za izhod iz
ročnega preverjanja vnesi `1234`.

Vsak zagon se shrani v:

- `saved_models/` kot datoteka `.pth`,
- `model_runs.json` skupaj z nastavitvami in končno natančnostjo.

Pri vsakem zagonu se datoteka `data.csv` na novo ustvari.

### Več naključnih zagonov

Za izvedbo več naključnih eksperimentov zaženi:

```bash
python random_runs.py
```

Program bo vprašal za število modelov in za vsak model naključno izbral
število podatkov, epoh ter odstotka za treniranje in validacijo.

### Statistična analiza

Ko imaš dovolj zapisov v `model_runs.json`, zaženi:

```bash
python statistic.py
```

Program prikaže:

- povezavo med natančnostjo in posameznimi dejavniki,
- standardizirane regresijske koeficiente,
- toplotno karto za vsaj 50 zagonov.

Grafi se prikažejo v oknih Matplotlib.

## Struktura projekta

| Datoteka | Namen |
| --- | --- |
| `main.py` | Treniranje, testiranje in ročno napovedovanje |
| `data_creation.py` | Ustvarjanje umetnih učnih podatkov |
| `random_runs.py` | Izvajanje več naključnih eksperimentov |
| `statistic.py` | Analiza in vizualizacija rezultatov |
| `requirements.txt` | Python odvisnosti projekta |

## Opomba

Projekt je namenjen učenju in eksperimentiranju. Ker uporablja umetno
generirane podatke in preprost model, napovedi niso namenjene dejanskemu
ocenjevanju učencev.
