from fastapi import FastAPI

app = FastAPI()

indian_places = {
    "Historical_and_Cultural_Landmarks": [
        "Taj Mahal (Agra, Uttar Pradesh)",
        "Qutub Minar (Delhi)",
        "Red Fort (Delhi)",
        "Ajanta and Ellora Caves (Maharashtra)",
        "Golden Temple (Amritsar, Punjab)",
    ],
    "Natural_Wonders": [
        "Himalayas",
        "Thar Desert (Rajasthan)",
        "Backwaters (Kerala)",
        "Sundarbans (West Bengal)",
    ],
    "Pilgrimage_Sites": [
        "Varanasi (Uttar Pradesh)",
        "Rishikesh (Uttarakhand)",
        "Vaishno Devi (Jammu and Kashmir)",
        "Haridwar (Uttarakhand)",
    ],
    "Hill_Stations": [
        "Shimla (Himachal Pradesh)",
        "Ooty (Tamil Nadu)",
        "Darjeeling (West Bengal)",
    ],
    "National_Parks_and_Wildlife_Sanctuaries": [
        "Jim Corbett National Park (Uttarakhand)",
        "Kaziranga National Park (Assam)",
        "Ranthambore National Park (Rajasthan)",
    ],
}


@app.get("/get_items/{name}")
async def hello(name):
    return indian_places.get(name)
