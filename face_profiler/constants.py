MIN_FACE_PX = 40

AGE_RANGES = [
    (0, 2, "0-2"), (3, 5, "3-5"), (6, 12, "6-12"), (13, 17, "13-17"),
    (18, 25, "18-25"), (26, 35, "26-35"), (36, 45, "36-45"),
    (46, 55, "46-55"), (56, 65, "56-65"), (66, 120, "66+"),
]

FAIRFACE_RACE_LABELS = [
    "White", "Black", "Latino_Hispanic", "East Asian",
    "Southeast Asian", "Indian", "Middle Eastern",
]


def age_to_range(age_float):
    age = int(round(age_float))
    for lo, hi, label in AGE_RANGES:
        if lo <= age <= hi:
            return label
    return f"{age}"
