import csv
from pathlib import Path
from uuid import uuid4


class DataCollector:
    # Колонки майбутнього датасету
    FIELDNAMES = [
        "session_id",
        "step",
        "lines_cleared",
        "holes",
        "bumpiness",
        "aggregate_height",
        "max_height",
        "landing_height",
        "score",
        "action",
    ]

    def __init__(self, filename="tetris_data.csv"):
        self.filename = Path(filename)

        # Створюємо окремий ідентифікатор запуску гри
        self.session_id = uuid4().hex[:8]

        self.step = 0

    def collect(
        self,
        lines_cleared,
        holes,
        bumpiness,
        aggregate_height,
        max_height,
        landing_height,
        score,
        action,
    ):
        self.step += 1

        row = {
            "session_id": self.session_id,
            "step": self.step,
            "lines_cleared": lines_cleared,
            "holes": holes,
            "bumpiness": bumpiness,
            "aggregate_height": aggregate_height,
            "max_height": max_height,
            "landing_height": landing_height,
            "score": score,
            "action": action,
        }

        self.save(row)

    def save(self, row):
        file_exists = (
            self.filename.exists()
            and self.filename.stat().st_size > 0
        )

        with self.filename.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)


def generate_sample_data(filename="tetris_data.csv", n_samples=250):
    """Генерує початковий набір даних для навчання, якщо гравцю потрібен готовий датасет."""
    import random

    collector = DataCollector(filename)
    print(f"Генерація {n_samples} тренувальних прикладів у {filename}...")

    for _ in range(n_samples):
        # 55% хороших ходів, 45% субоптимальних/поганих ходів
        is_good = random.random() < 0.55

        if is_good:
            lines_cleared = random.choices([0, 1, 2, 3, 4], weights=[0.45, 0.35, 0.12, 0.05, 0.03])[0]
            holes = random.choices([0, 1], weights=[0.85, 0.15])[0]
            bumpiness = random.randint(1, 8)
            aggregate_height = random.randint(4, 35)
            max_height = max(1, aggregate_height // random.randint(3, 7))
            landing_height = random.randint(1, max_height + 2)
            action = "good"
        else:
            lines_cleared = 0 if random.random() < 0.9 else 1
            holes = random.randint(1, 6)
            bumpiness = random.randint(6, 18)
            aggregate_height = random.randint(25, 75)
            max_height = random.randint(6, 18)
            landing_height = random.randint(max(2, max_height - 2), min(20, max_height + 4))
            action = "bad"

        score = random.randint(0, 3000)

        collector.collect(
            lines_cleared=lines_cleared,
            holes=holes,
            bumpiness=bumpiness,
            aggregate_height=aggregate_height,
            max_height=max_height,
            landing_height=landing_height,
            score=score,
            action=action,
        )

    print(f"Готово! Записано {collector.step} рядків.")


if __name__ == "__main__":
    generate_sample_data()
