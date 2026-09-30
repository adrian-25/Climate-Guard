from src.cities import CITIES


def test_city_registry_contains_only_validated_active_model_cities():
    assert list(CITIES) == ["delhi", "lucknow", "nagpur", "ahmedabad", "mumbai"]
    for city in CITIES.values():
        assert city["model_status"] == "validated"
        assert {"name", "state", "region", "region_type", "lat", "lon"} <= city.keys()
