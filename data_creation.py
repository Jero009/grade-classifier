import random as r
import csv

def create_data(num_rows):
    rows = []
    for i in range(num_rows):
        percent = round(r.random() * 100, 1)
        if percent <49:
            grade = 1
        elif percent < 62:
            grade = 2
        elif percent < 75:
            grade = 3
        elif percent < 89:
            grade = 4
        else:
            grade = 5
        
        rows.append((percent, grade))

    with open("data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["percent", "grade"])   # header
        w.writerows(rows)  
