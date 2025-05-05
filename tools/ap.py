stats = [  # (class, gts, ap)
    ("Boeing737", 4208, 0.540),
    ("Boeing777", 1660, 0.428),
    ("Boeing747", 1717, 0.871),
    ("Boeing787", 1757, 0.604),
    ("A321", 2702, 0.764),
    ("A220", 6485, 0.617),
    ("A330", 1693, 0.632),
    ("A350", 1122, 0.689),
    ("C919", 138, 0.000),
    ("ARJ21", 186, 0.493),
    ("other-airplane", 10413, 0.823),
    ("Passenger_Ship", 840, 0.303),
    ("Motorboat", 9263, 0.502),
    ("Fishing_Boat", 8392, 0.467),
    ("Tugboat", 2338, 0.681),
    ("Engineering_Ship", 1955, 0.590),
    ("Liquid_Cargo_Ship", 3431, 0.694),
    ("Dry_Cargo_Ship", 12017, 0.759),
    ("Warship", 782, 0.668),
    ("other-ship", 2848, 0.181),
    ("Small_Car", 143231, 0.633),
    ("Bus", 1018, 0.273),
    ("Cargo_Truck", 9247, 0.531),
    ("Dump_Truck", 25762, 0.528),
    ("Van", 132259, 0.610),
    ("Trailer", 586, 0.045),
    ("Tractor", 262, 0.100),
    ("Truck_Tractor", 923, 0.018),
    ("Excavator", 890, 0.212),
    ("other-vehicle", 3062, 0.073),
    ("Baseball_Field", 1061, 0.887),
    ("Basketball_Court", 1321, 0.813),
    ("Football_Field", 880, 0.820),
    ("Tennis_Court", 2941, 0.908),
    ("Roundabout", 584, 0.858),
    ("Intersection", 6953, 0.665),
    ("Bridge", 1212, 0.424),
]

import math
eps = 1e-6
median_gts = sorted(g for _, g, _ in stats)[len(stats)//2]

# 计算 raw 权重
raw_w = []
for cls, gts, ap in stats:
    rarity = median_gts / (gts + eps)
    difficulty = 1 - ap
    raw_w.append(rarity * difficulty)

# 归一化，使平均值 = 1，并做 0.2~5 的截断（可调）
mean_w = sum(raw_w) / len(raw_w)
weights = [max(0.8, min(2, round(w / mean_w, 3))) for w in raw_w]

# 打印
for (cls, _, _), w in zip(stats, weights):
    print(f"{cls:20s}: {w}")
print("\nclass_weight =", weights)