import random

import numpy as np
import torch


def set_all_seeds(seed):
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def source_labels(dataset):
    if dataset == "Office":
        label = [
            "back pack", "bike", "bike helmet", "bookcase", "bottle", "calculator",
            "desk chair", "desk lamp", "desktop computer", "file cabinet", "headphones",
            "keyboard", "laptop computer", "letter tray", "mobile phone", "monitor",
            "mouse", "mug", "paper notebook", "pen", "phone", "printer", "projector",
            "punchers", "ring binder", "ruler", "scissors", "speaker", "stapler",
            "tape dispenser", "trash can",
        ]
    elif dataset == "OfficeHome":
        label = [
            "Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike",
            "Bottle", "Bucket", "Calculator", "Calendar", "Candles",
            "Chair", "Clipboards", "Computer", "Couch", "Curtains",
            "Desk_Lamp", "Drill", "Eraser", "Exit_Sign", "Fan",
            "File_Cabinet", "Flipflops", "Flowers", "Folder", "Fork",
            "Glasses", "Hammer", "Helmet", "Kettle", "Keyboard",
            "Knives", "Lamp_Shade", "Laptop", "Marker", "Monitor",
            "Mop", "Mouse", "Mug", "Notebook", "Oven",
            "Pan", "Paper_Clip", "Pen", "Pencil", "Postit_Notes",
            "Printer", "Push_Pin", "Radio", "Refrigerator", "Ruler",
            "Scissors", "Screwdriver", "Shelf", "Sink", "Sneakers",
            "Soda", "Speaker", "Spoon", "Table", "Telephone",
            "ToothBrush", "Toys", "Trash_Can", "TV", "Webcam",
        ]
    elif dataset == "VisDA":
        label = [
            "aeroplane", "bicycle", "bus", "car", "horse",
            "knife", "motorcycle", "person", "plant", "skateboard",
            "train", "truck",
        ]
    elif dataset == "DomainNet":
        label = [
            "aircraft_carrier", "airplane", "alarm_clock", "ambulance", "angel",
            "animal_migration", "ant", "anvil", "apple", "arm", "asparagus", "axe",
            "backpack", "banana", "bandage", "barn", "baseball", "baseball_bat",
            "basket", "basketball", "bat", "bathtub", "beach", "bear", "beard", "bed",
            "bee", "belt", "bench", "bicycle", "binoculars", "bird", "birthday_cake",
            "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie",
            "bracelet", "brain", "bread", "bridge", "broccoli", "broom", "bucket",
            "bulldozer", "bus", "bush", "butterfly", "cactus", "cake", "calculator",
            "calendar", "camel", "camera", "camouflage", "campfire", "candle", "cannon",
            "canoe", "car", "carrot", "castle", "cat", "ceiling_fan", "cello",
            "cell_phone", "chair", "chandelier", "church", "circle", "clarinet", "clock",
            "cloud", "coffee_cup", "compass", "computer", "cookie", "cooler", "couch", "cow",
            "crab", "crayon", "crocodile", "crown", "cruise_ship", "cup", "diamond",
            "dishwasher", "diving_board", "dog", "dolphin", "donut", "door", "dragon",
            "dresser", "drill", "drums", "duck", "dumbbell", "ear", "elbow", "elephant",
            "envelope", "eraser", "eye", "eyeglasses", "face", "fan", "feather", "fence",
            "finger", "fireplace", "firetruck", "fire_hydrant", "fish", "flamingo",
            "flashlight", "flip_flops", "floor_lamp", "flower", "flying_saucer", "foot",
            "fork", "frog", "frying_pan", "garden", "garden_hose", "giraffe", "goatee",
            "golf_club", "grapes", "grass", "guitar", "hamburger", "hammer", "hand", "harp",
            "hat", "headphones", "hedgehog", "helicopter", "helmet", "hexagon", "hockey_puck",
            "hockey_stick", "horse", "hospital", "hot_air_balloon", "hot_dog", "hot_tub",
            "hourglass", "house", "house_plant", "hurricane", "ice_cream", "jacket", "jail",
            "kangaroo", "key", "keyboard", "knee", "knife", "ladder", "lantern", "laptop",
            "leaf", "leg", "lighter", "lighthouse", "lightning", "light_bulb", "line", "lion",
            "lipstick", "lobster", "lollipop", "mailbox", "map", "marker", "matches", "megaphone",
            "mermaid", "microphone", "microwave", "monkey", "moon", "mosquito", "motorbike",
            "mountain", "mouse", "moustache", "mouth", "mug", "mushroom", "nail", "necklace",
            "nose", "ocean", "octagon", "octopus", "onion", "oven", "owl", "paintbrush",
            "paint_can", "palm_tree", "panda", "pants", "paper_clip", "parachute", "parrot",
            "passport", "peanut", "pear", "peas", "pencil", "penguin", "piano", "pickup_truck",
            "picture_frame", "pig", "pillow", "pineapple", "pizza", "pliers", "police_car",
            "pond", "pool", "popsicle", "postcard", "potato", "power_outlet", "purse", "rabbit",
            "raccoon", "radio", "rain", "rainbow", "rake", "remote_control", "rhinoceros",
            "rifle", "river", "rollerskates", "roller_coaster", "sailboat", "sandwich", "saw",
            "saxophone", "school_bus", "scissors", "scorpion", "screwdriver", "sea_turtle",
            "see_saw", "shark", "sheep", "shoe", "shorts", "shovel", "sink", "skateboard",
            "skull", "skyscraper", "sleeping_bag", "smiley_face", "snail", "snake", "snorkel",
            "snowflake", "snowman", "soccer_ball", "sock", "speedboat", "spider", "spoon",
            "spreadsheet", "square", "squiggle", "squirrel", "stairs", "star", "steak", "stereo",
            "stethoscope", "stitches", "stop_sign", "stove", "strawberry", "streetlight",
            "string_bean", "submarine", "suitcase", "sun", "swan", "sweater", "swing_set",
            "sword", "syringe", "t-shirt", "table", "teapot", "teddy-bear", "telephone",
            "television", "tennis_racquet", "tent", "The_Eiffel_Tower",
            "The_Great_Wall_of_China", "The_Mona_Lisa", "tiger", "toaster", "toe", "toilet",
            "tooth", "toothbrush", "toothpaste", "tornado", "tractor", "traffic_light", "train",
            "tree", "triangle", "trombone", "truck", "trumpet", "umbrella", "underwear", "van",
            "vase", "violin", "washing_machine", "watermelon", "waterslide", "whale", "wheel",
            "windmill", "wine_bottle", "wine_glass", "wristwatch", "yoga", "zebra", "zigzag",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return [name.replace("_", " ").lower() for name in label]


SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}.",
)


DATASET_SPLITS = {
    "Office": (10, 10, 31),
    "OfficeHome": (10, 5, 65),
    "VisDA": (6, 6, 12),
    "DomainNet": (150, 50, 345),
}


DATASET_DOMAINS = {
    "Office": ["amazon", "dslr", "webcam"],
    "OfficeHome": ["Art", "Clipart", "Product", "RealWorld"],
    "VisDA": ["train", "validation"],
    "DomainNet": ["painting", "real", "sketch"],
}
