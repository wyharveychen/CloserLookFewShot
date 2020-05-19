# Create train, val, test split
from generate_cross_domain_split import generate_cross_domain_split
base_classes = ["beaver", "dolphin", "otter"]
val_classes = ["seal"]
novel_classes = ["whale"]
generate_cross_domain_split(base_classes, val_classes, novel_classes)
