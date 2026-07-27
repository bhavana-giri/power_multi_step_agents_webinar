"""Static vehicle catalog for the Redis Motors shopping assistant."""

CATALOG = [
    {"model": "Toyota RAV4 Hybrid", "type": "SUV", "fuel": "hybrid", "price_usd": 32000, "mpg": 40, "seats": 5},
    {"model": "Honda CR-V Hybrid", "type": "SUV", "fuel": "hybrid", "price_usd": 34000, "mpg": 38, "seats": 5},
    {"model": "Hyundai Tucson Hybrid", "type": "SUV", "fuel": "hybrid", "price_usd": 33000, "mpg": 38, "seats": 5},
    {"model": "Ford Escape Hybrid", "type": "SUV", "fuel": "hybrid", "price_usd": 31000, "mpg": 39, "seats": 5},
    {"model": "Kia Sportage Hybrid", "type": "SUV", "fuel": "hybrid", "price_usd": 30500, "mpg": 43, "seats": 5},
    {"model": "Toyota Highlander Hybrid", "type": "SUV", "fuel": "hybrid", "price_usd": 41000, "mpg": 35, "seats": 8},
    {"model": "Tesla Model Y", "type": "SUV", "fuel": "electric", "price_usd": 44000, "mpg": 0, "seats": 5},
    {"model": "Mazda CX-50", "type": "SUV", "fuel": "gas", "price_usd": 29000, "mpg": 27, "seats": 5},
    {"model": "Toyota Camry Hybrid", "type": "sedan", "fuel": "hybrid", "price_usd": 29500, "mpg": 51, "seats": 5},
    {"model": "Honda Civic", "type": "sedan", "fuel": "gas", "price_usd": 25000, "mpg": 35, "seats": 5},
]


def catalog_prompt_block() -> str:
    lines = [
        f"- {c['model']} ({c['type']}, {c['fuel']}, ${c['price_usd']:,}, "
        f"{c['mpg']} mpg, {c['seats']} seats)"
        for c in CATALOG
    ]
    return "Current inventory:\n" + "\n".join(lines)
