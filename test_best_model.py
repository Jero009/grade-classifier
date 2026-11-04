import json
import random
from pathlib import Path

import torch
import torch.nn as nn


MODEL_HISTORY = Path(__file__).with_name("model_runs.json")
PROJECT_DIR = Path(__file__).parent
BEST_MODEL_PATH = PROJECT_DIR / "best_model" / "best_model.pth"


def create_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 7),
        nn.ReLU(),
        nn.Linear(7, 5),
    )


def load_best_model() -> tuple[nn.Module, dict]:
    with MODEL_HISTORY.open(encoding="utf-8") as file:
        runs = json.load(file)

    if not BEST_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Best model was not found at {BEST_MODEL_PATH}.")

    best_run = max(runs, key=lambda run: run["final_accuracy"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model().to(device)
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, best_run


def expected_grade(percent: float) -> int:
    if percent < 49:
        return 1
    if percent < 62:
        return 2
    if percent < 75:
        return 3
    if percent < 89:
        return 4
    return 5


def test_random_data(model: nn.Module, device: torch.device) -> None:
    while True:
        try:
            count = int(input("How many new random examples should be tested? "))
            if count > 0:
                break
        except ValueError:
            pass
        print("Please enter a positive whole number.")

    percentages = [round(random.uniform(0, 100), 1) for _ in range(count)]
    expected = torch.tensor(
        [expected_grade(percent) - 1 for percent in percentages],
        dtype=torch.long,
        device=device,
    )
    inputs = torch.tensor(
        [[percent / 100] for percent in percentages],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        predictions = torch.argmax(model(inputs), dim=1)

    correct = int((predictions == expected).sum().item())
    accuracy = correct / count * 100
    print(f"Random unseen-data accuracy: {accuracy:.2f}% ({correct}/{count})")

    errors = [
        (percent, prediction.item() + 1, actual.item() + 1)
        for percent, prediction, actual in zip(percentages, predictions, expected)
        if prediction != actual
    ]
    for percent, prediction, actual in errors[:10]:
        print(f"  {percent:g}% -> predicted {prediction}, expected {actual}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")


def main() -> None:
    model, run = load_best_model()
    device = next(model.parameters()).device

    print(f"Loaded best model: {BEST_MODEL_PATH}")
    print(f"Recorded test accuracy: {run['final_accuracy']:.2f}%")
    print("Enter a percentage to test it manually.")
    print("Enter 'r' to test new random data or 'q' to exit.")

    while True:
        choice = input("\nPercentage, 'r', or 'q': ").strip().lower()
        if choice == "q" or choice == "1234":
            break
        if choice == "r":
            test_random_data(model, device)
            continue

        try:
            percent = float(choice)
        except ValueError:
            print("Please enter a percentage, 'r', or 'q'.")
            continue

        if not 0 <= percent <= 100:
            print("Please enter a percentage between 0 and 100.")
            continue

        with torch.no_grad():
            input_tensor = torch.tensor([[percent / 100]], dtype=torch.float32, device=device)
            probabilities = torch.softmax(model(input_tensor), dim=1)[0]
            prediction = int(torch.argmax(probabilities).item()) + 1

        print(f"Predicted grade for {percent:g}%: {prediction}")
        for grade, probability in enumerate(probabilities, start=1):
            print(f"Grade {grade}: {probability.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
