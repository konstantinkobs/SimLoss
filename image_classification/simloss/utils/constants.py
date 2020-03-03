from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

CIFAR_PATH = PROJECT_DIR / 'data' / 'external'
SIM_MATRIX_PATH = PROJECT_DIR / 'data' / 'processed' / 'sim_matrix.npy'

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR100_SUPERCLASSES_INV = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
    'food containers': ['bottles', 'bowls', 'cans', 'cups', 'plates'],
    'fruit and vegetables': ['apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers'],
    'household electrical devices': ['clock', 'computer keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup truck', 'train'],
    'vehicles 2': ['lawnmower', 'rocket', 'streetcar', 'tank', 'tractor']
}

# Note: We removed classes which are not in the word embedding vocabulary
CIFAR100_BLACKLIST = ['aquarium fish', 'sweet peppers', 'computer keyboard', 'pickup truck']

CIFAR100_SUPERCLASSES = {c: sc for sc in CIFAR100_SUPERCLASSES_INV for c in CIFAR100_SUPERCLASSES_INV[sc]}
CIFAR100_CLASSES = sorted(CIFAR100_SUPERCLASSES.keys())
CIFAR100_CLASSES_FILTERED = [c for c in CIFAR100_CLASSES if c not in CIFAR100_BLACKLIST]
print(CIFAR100_CLASSES)
