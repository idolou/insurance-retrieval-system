import httpx
import datetime
from langchain_core.tools import tool


@tool
def get_historical_weather(city: str, date: str) -> str:
    """
    Get the historical weather for a specific city and date.
    Use this tool when the date is in the past to verify claim conditions.

    Args:
        city: The name of the city (e.g., "Austin").
        date: The date in YYYY-MM-DD format (e.g., "2024-11-16").
    """
    try:
        # 1. Geocode the city
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city, "count": 1, "language": "en", "format": "json"}
        response = httpx.get(geocode_url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            return f"Could not find location: {city}"

        location = data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]

        # 2. Check if date is in the future (simple check)
        target_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        today = datetime.datetime.now().date()

        if target_date > today:
            return "Date is in the future. Cannot retrieve historical weather."

        # 3. Fetch Historical Weather
        # using archive API for past dates
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date,
            "end_date": date,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,wind_speed_10m_max",
            "timezone": "auto",
        }

        response = httpx.get(weather_url, params=params)
        response.raise_for_status()
        weather_data = response.json()

        if "daily" not in weather_data:
            return f"No daily data found for {city} on {date}"

        daily = weather_data["daily"]

        report = (
            f"Weather Report for {city} on {date}:\n"
            f"- Max Temp: {daily['temperature_2m_max'][0]} {weather_data['daily_units']['temperature_2m_max']}\n"
            f"- Min Temp: {daily['temperature_2m_min'][0]} {weather_data['daily_units']['temperature_2m_min']}\n"
            f"- Precipitation: {daily['precipitation_sum'][0]} {weather_data['daily_units']['precipitation_sum']}\n"
            f"- Rain: {daily['rain_sum'][0]} {weather_data['daily_units']['rain_sum']}\n"
            f"- Snow: {daily['snowfall_sum'][0]} {weather_data['daily_units']['snowfall_sum']}\n"
            f"- Max Wind Speed: {daily['wind_speed_10m_max'][0]} {weather_data['daily_units']['wind_speed_10m_max']}\n"
        )

        return report

    except Exception as e:
        return f"Error fetching weather: {str(e)}"
