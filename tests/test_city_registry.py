from src.cities import CITIES, CITY_REGISTRY

INDIA_KEYS = ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
EUROPE_KEYS = [
    "madrid", "seville", "valencia", "barcelona", "zaragoza", "bilbao",
    "lisbon", "porto", "evora", "andorra_la_vella", "monaco",
]
VALID_STATUSES = {"validated", "europe-v1"}


def test_india_cities_present_and_validated():
    """The five original India cities must still be present."""
    for key in INDIA_KEYS:
        assert key in CITIES, f"India city missing: {key}"
        assert CITY_REGISTRY[key]["model_status"] == "validated"


def test_europe_cities_present():
    """All 11 Europe cities must be registered."""
    for key in EUROPE_KEYS:
        assert key in CITIES, f"Europe city missing: {key}"
        assert CITY_REGISTRY[key]["dataset_region"] == "europe"


def test_all_registry_entries_have_required_fields():
    """Every city in the registry must have the mandatory fields."""
    for city in CITY_REGISTRY.values():
        assert {"name", "lat", "lon", "region_type", "dataset_region"} <= city.keys()
        assert city["model_status"] in VALID_STATUSES


def test_model_status_stripped_from_public_cities():
    """Public CITIES dict must not expose internal model_status."""
    assert all("model_status" not in city for city in CITIES.values())


def test_region_routing():
    """dataset_region field routes cities to the correct model."""
    from src.cities import CITIES
    india  = {k for k, v in CITIES.items() if v.get("dataset_region") == "india"}
    europe = {k for k, v in CITIES.items() if v.get("dataset_region") == "europe"}
    assert india  == set(INDIA_KEYS)
    assert europe == set(EUROPE_KEYS)
