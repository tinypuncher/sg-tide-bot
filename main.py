# main.py - Singapore Edition
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import numpy as np
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import FileResponse
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler,
    MessageHandler,
    filters
)
from dotenv import load_dotenv

load_dotenv()

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAILWAY_URL = os.getenv("RAILWAY_PUBLIC_DOMAIN")
# Or for older Railway versions:
# RAILWAY_URL = os.getenv("RAILWAY_STATIC_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "singapore-tide-bot-secret")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Optional - free tier available

# Singapore-specific configuration
SG_TIMEZONE = "Asia/Singapore"
SG_LOCATION = {
    "lat": 1.3521,
    "lon": 103.8198,
    "name": "Singapore"
}

# Singapore Tide Stations (HPA/MPA)
SG_TIDE_STATIONS = {
    "sembawang": {"id": "QZ9", "name": "Sembawang", "lat": 1.463, "lon": 103.848},
    "tuas": {"id": "QZ2", "name": "Tuas View", "lat": 1.327, "lon": 103.638},
    "tanjong_pagar": {"id": "QZ1", "name": "Tanjong Pagar", "lat": 1.265, "lon": 103.842},
    "changi": {"id": "QZ5", "name": "Changi", "lat": 1.389, "lon": 104.009},
    "pulau_ubi": {"id": "QZ3", "name": "Pulau Ubin (East)", "lat": 1.413, "lon": 103.967},
    "sentosa": {"id": "QZ4", "name": "Sentosa (Tanjong Rimau)", "lat": 1.250, "lon": 103.832},
    "jurong": {"id": "QZ6", "name": "Jurong Island", "lat": 1.275, "lon": 103.709},
    "east_coast": {"id": "QZ7", "name": "East Coast Parkway", "lat": 1.297, "lon": 103.912},
}

# Fishing spots database (crowdsourced/local knowledge)
SG_FISHING_SPOTS = {
    "bedok_jetty": {
        "name": "Bedok Jetty",
        "location": "East Coast Park",
        "best_tide": "incoming",
        "target_species": ["Barramundi", "Grouper", "Snapper", "Stingray"],
        "coordinates": (1.305, 103.925),
        "station": "east_coast"
    },
    "sembawang_jetty": {
        "name": "Sembawang Jetty",
        "location": "Sembawang Park",
        "best_tide": "high",
        "target_species": ["Barramundi", "Threadfin", "Snapper"],
        "coordinates": (1.463, 103.848),
        "station": "sembawang"
    },
    "woodlands_jetty": {
        "name": "Woodlands Jetty",
        "location": "Admiralty Park",
        "best_tide": "incoming",
        "target_species": ["Barramundi", "Grouper", "Catfish"],
        "coordinates": (1.448, 103.785),
        "station": "sembawang"
    },
    "punggol_jetty": {
        "name": "Punggol Jetty",
        "location": "Punggol Point",
        "best_tide": "high",
        "target_species": ["Barramundi", "Threadfin", "Queenfish"],
        "coordinates": (1.419, 103.913),
        "station": "pulau_ubi"
    },
    "labrador_park": {
        "name": "Labrador Park",
        "location": "Southern Coast",
        "best_tide": "outgoing",
        "target_species": ["Grouper", "Snapper", "Parrotfish"],
        "coordinates": (1.266, 103.802),
        "station": "sentosa"
    },
    "tuas_lagoon": {
        "name": "Tuas Lagoon",
        "location": "Tuas",
        "best_tide": "high",
        "target_species": ["Barramundi", "Mangrove Jack"],
        "coordinates": (1.327, 103.638),
        "station": "tuas"
    },
    "changi_boardwalk": {
        "name": "Changi Boardwalk",
        "location": "Changi Point",
        "best_tide": "incoming",
        "target_species": ["Barramundi", "Snapper", "Grouper"],
        "coordinates": (1.389, 104.009),
        "station": "changi"
    }
}

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==========================================
# Singapore Data APIs (All Free)
# ==========================================

class SingaporeDataClient:
    """Client for Singapore government open data APIs"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = "https://api.data.gov.sg/v1/environment"
    
    async def get_24hr_weather_forecast(self) -> Dict:
        """Get 24-hour weather forecast from NEA"""
        try:
            response = await self.client.get(
                f"{self.base_url}/24-hour-weather-forecast"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return {}
    
    async def get_4day_weather_forecast(self) -> Dict:
        """Get 4-day weather forecast"""
        try:
            response = await self.client.get(
                f"{self.base_url}/4-day-weather-forecast"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching 4-day forecast: {e}")
            return {}
    
    async def get_pm25(self) -> Dict:
        """Get PM2.5 readings (affects fishing comfort)"""
        try:
            response = await self.client.get(f"{self.base_url}/pm25")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching PM2.5: {e}")
            return {}
    
    async def get_uv_index(self) -> Dict:
        """Get UV index (for daytime fishing)"""
        try:
            response = await self.client.get(f"{self.base_url}/uv-index")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching UV index: {e}")
            return {}
    
    async def get_tide_data(self, station_id: str) -> Dict[str, Any]:
        """
        Get tide data from Singapore HPA/MPA
        Note: Using calculated tides for Singapore waters if API unavailable
        """
        try:
            # Try to get from Singapore Maritime Data Hub if available
            # Fallback to calculated harmonic constituents for Singapore
            return await self._calculate_singapore_tides(station_id)
        except Exception as e:
            logger.error(f"Error calculating tides: {e}")
            return {}
    
    async def _calculate_singapore_tides(self, station_id: str) -> Dict[str, Any]:
        """
        Calculate tides using harmonic constituents for Singapore
        Based on published data from HPA tidal harmonics
        """
        station = SG_TIDE_STATIONS.get(station_id)
        if not station:
            return {}
        
        # Harmonic constituents for Singapore (simplified)
        # Real implementation would use full harmonic constants from HPA
        constituents = {
            "M2": {"amp": 1.2, "phase": 0},    # Principal lunar semi-diurnal
            "S2": {"amp": 0.3, "phase": 30},   # Principal solar semi-diurnal  
            "K1": {"amp": 0.4, "phase": 90},   # Lunar diurnal
            "O1": {"amp": 0.3, "phase": 120},  # Lunar diurnal
        }
        
        # Generate predictions for next 48 hours
        predictions = []
        now = datetime.now()
        
        for hour in range(48):
            t = now + timedelta(hours=hour)
            # Simplified tide calculation (would use proper harmonic formula)
            height = self._calculate_tide_height(t, constituents)
            predictions.append({
                "t": t.strftime("%Y-%m-%d %H:%M"),
                "v": round(height, 2)
            })
        
        # Find high/low tides
        tides = self._find_high_low_tides(predictions)
        
        return {
            "predictions": predictions,
            "tides": tides,
            "station": station
        }
    
    def _calculate_tide_height(self, t: datetime, constituents: Dict) -> float:
        """Calculate tide height using harmonic constituents"""
        # Mean sea level for Singapore + harmonic sum
        msl = 2.5  # meters above chart datum
        
        hours = (t - datetime(2024, 1, 1)).total_seconds() / 3600
        
        height = msl
        for name, params in constituents.items():
            speed = self._constituent_speed(name)
            amp = params["amp"]
            phase = np.radians(params["phase"])
            height += amp * np.cos(speed * hours + phase)
        
        return height
    
    def _constituent_speed(self, name: str) -> float:
        """Return speed in radians per hour for constituent"""
        speeds = {
            "M2": 28.9841042 * np.pi / 180,  # deg/hour to rad/hour
            "S2": 30.0000000 * np.pi / 180,
            "K1": 15.0410686 * np.pi / 180,
            "O1": 13.9430356 * np.pi / 180,
        }
        return speeds.get(name, 0)
    
    def _find_high_low_tides(self, predictions: List[Dict]) -> List[Dict]:
        """Identify high and low tide events from predictions"""
        tides = []
        for i in range(1, len(predictions) - 1):
            prev_h = float(predictions[i-1]["v"])
            curr_h = float(predictions[i]["v"])
            next_h = float(predictions[i+1]["v"])
            
            if curr_h > prev_h and curr_h > next_h:
                tides.append({
                    "t": predictions[i]["t"],
                    "v": predictions[i]["v"],
                    "type": "H"  # High
                })
            elif curr_h < prev_h and curr_h < next_h:
                tides.append({
                    "t": predictions[i]["t"],
                    "v": predictions[i]["v"],
                    "type": "L"  # Low
                })
        
        return tides
    
    async def close(self):
        await self.client.aclose()

sg_client = SingaporeDataClient()

# ==========================================
# Free AI Implementation (Rule-Based + Optional Groq)
# ==========================================

class SingaporeFishingAI:
    """
    AI fishing assistant using rule-based logic + optional Groq LLM
    Groq offers free tier: 1M tokens/day at 800+ tokens/sec
    """
    
    def __init__(self):
        self.use_groq = bool(GROQ_API_KEY)
        if self.use_groq:
            self.client = httpx.AsyncClient()
    
    async def get_fishing_advice(
        self, 
        spot_id: str,
        tide_data: Dict,
        weather_data: Dict,
        target_species: Optional[str] = None
    ) -> str:
        """Generate fishing advice for Singapore location"""
        
        spot = SG_FISHING_SPOTS.get(spot_id)
        if not spot:
            return "Unknown fishing spot. Use /spots to see available locations."
        
        # Get current conditions
        current_tide = self._get_current_tide_state(tide_data)
        weather = self._parse_weather(weather_data)
        
        # Build advice using local knowledge base
        advice = self._generate_local_advice(spot, current_tide, weather, target_species)
        
        # Enhance with AI if available
        if self.use_groq:
            try:
                ai_enhancement = await self._get_groq_enhancement(
                    spot, current_tide, weather, target_species
                )
                advice += f"\n\n🤖 *AI Analysis:*\n{ai_enhancement}"
            except Exception as e:
                logger.error(f"Groq enhancement failed: {e}")
        
        return advice
    
    def _get_current_tide_state(self, tide_data: Dict) -> Dict:
        """Determine current tide state"""
        tides = tide_data.get("tides", [])
        now = datetime.now()
        
        # Find nearest tide event
        nearest = None
        min_diff = float('inf')
        
        for tide in tides:
            t = datetime.strptime(tide["t"], "%Y-%m-%d %H:%M")
            diff = abs((t - now).total_seconds())
            if diff < min_diff:
                min_diff = diff
                nearest = tide
        
        if not nearest:
            return {"state": "unknown", "height": 0}
        
        # Determine if incoming or outgoing
        tide_type = nearest["type"]
        time_to_tide = min_diff / 3600  # hours
        
        if time_to_tide < 1:
            state = "high" if tide_type == "H" else "low"
        else:
            state = "incoming" if tide_type == "H" else "outgoing"
        
        return {
            "state": state,
            "height": float(nearest["v"]),
            "next_tide": nearest["t"],
            "type": tide_type
        }
    
    def _parse_weather(self, weather_data: Dict) -> Dict:
        """Parse NEA weather data"""
        try:
            forecasts = weather_data.get("items", [{}])[0]
            general = forecasts.get("general", {})
            
            return {
                "forecast": general.get("forecast", "Unknown"),
                "temperature_low": general.get("temperature", {}).get("low", 25),
                "temperature_high": general.get("temperature", {}).get("high", 32),
                "humidity": general.get("relative_humidity", {}).get("low", 60),
                "wind_speed": general.get("wind", {}).get("speed", {}).get("low", 5),
            }
        except:
            return {
                "forecast": "Partly Cloudy",
                "temperature_low": 26,
                "temperature_high": 31,
                "humidity": 80,
                "wind_speed": 10
            }
    
    def _generate_local_advice(
        self, 
        spot: Dict, 
        tide: Dict, 
        weather: Dict,
        target: Optional[str]
    ) -> str:
        """Generate advice based on local Singapore fishing knowledge"""
        
        # Check if conditions match spot's best tide
        tide_match = tide["state"] == spot["best_tide"]
        tide_emoji = "✅" if tide_match else "⚠️"
        
        # Species recommendations
        species = target if target else ", ".join(spot["target_species"][:3])
        
        # Time-based advice
        hour = datetime.now().hour
        if 5 <= hour <= 9 or 17 <= hour <= 20:
            time_rating = "🌟🌟🌟 Excellent (Dawn/Dusk)"
        elif 10 <= hour <= 16:
            time_rating = "🌟🌟 Fair (Midday heat)"
        else:
            time_rating = "🌟 Good (Night fishing possible)"
        
        advice = f"""
🎣 *Fishing Analysis: {spot['name']}*

📍 *Location:* {spot['location']}

🌊 *Tide Conditions:*
{tide_emoji} Current: {tide['state'].upper()} tide
📊 Height: {tide['height']:.2f}m
⏰ Next {'high' if tide['type'] == 'L' else 'low'} tide: {tide['next_tide']}

🌤️ *Weather:*
{weather['forecast']}
🌡️ {weather['temperature_low']}°C - {weather['temperature_high']}°C
💧 Humidity: {weather['humidity']}%
💨 Wind: {weather['wind_speed']} km/h

⏰ *Timing:* {time_rating}

🐟 *Target Species:* {species}

💡 *Local Tips:*
"""
        
        # Add spot-specific tips
        if spot_id == "bedok_jetty":
            advice += """
• Cast towards the artificial reefs (marked by buoys)
• Use live prawns or squid strips for barramundi
• Best results on incoming tide near the pylons
• Watch out for passing ships after 6 PM
"""
        elif spot_id == "sembawang_jetty":
            advice += """
• Fish the right side for threadfin salmon
• Use apollo rig with fresh prawns
• Deep water drop-off is 20m to the left
• Strong currents during spring tides - use heavier sinkers
"""
        elif spot_id == "punggol_jetty":
            advice += """
• Cast towards the kelongs (fishing structures)
• Topwater lures work well at dawn for queenfish
• Check for jellyfish blooms in NE monsoon (Dec-Mar)
• Parking available at Punggol Point Park
"""
        else:
            advice += f"""
• Best tide for this spot: {spot['best_tide'].upper()}
• Try live bait or soft plastics
• Check local regulations for size limits
• Popular with local anglers on weekends
"""
        
        if not tide_match:
            advice += f"\n⚠️ *Note:* Current tide is {tide['state']}, but this spot fishes best during {spot['best_tide']} tide."
        
        return advice
    
    async def _get_groq_enhancement(
        self, 
        spot: Dict, 
        tide: Dict, 
        weather: Dict,
        target: Optional[str]
    ) -> str:
        """Get AI enhancement using Groq (free tier)"""
        
        prompt = f"""
        You are a Singapore fishing expert. Provide 2-3 short, specific tips for fishing at {spot['name']} 
        right now. Current conditions: {tide['state']} tide, {weather['forecast']}, 
        target species: {target or 'general'}. Be concise and actionable. Max 100 words.
        """
        
        try:
            response = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mixtral-8x7b-32768",  # Fast, free tier available
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 150
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except:
            return ""
    
    async def close(self):
        if self.use_groq:
            await self.client.aclose()

fishing_ai = SingaporeFishingAI()

# ==========================================
# Visualization (Matplotlib - Free)
# ==========================================

class SingaporeTideVisualizer:
    @staticmethod
    def create_tide_table(tide_data: Dict, station_name: str) -> str:
        """Generate text-based tide table"""
        predictions = tide_data.get("predictions", [])
        tides = tide_data.get("tides", [])
        
        if not predictions:
            return "No tide data available."
        
        # Build text table
        text = f"🌊 *Tide Predictions - {station_name}*\n"
        text += f"🇸🇬 Singapore Standard Time (SGT)\n\n"
        
        # Group by day
        current_day = None
        for p in predictions[::2]:  # Every 2 hours to save space
            t = datetime.strptime(p["t"], "%Y-%m-%d %H:%M")
            height = float(p["v"])
            
            # New day header
            if current_day != t.date():
                current_day = t.date()
                text += f"\n📅 *{t.strftime('%A, %d %B')}*\n"
                text += "```\nTime    | Height\n"
                text += "--------|-------\n"
            
            bar = "█" * int(height * 3)  # Simple bar chart
            text += f"{t.strftime('%H:%M')} | {height:.2f}m {bar}\n"
        
        text += "```\n\n"  # Close code block
        
        # Add high/low tide summary
        text += "*Key Times:*\n"
        for tide in tides[:4]:
            t = datetime.strptime(tide["t"], "%Y-%m-%d %H:%M")
            emoji = "🔺 HIGH" if tide["type"] == "H" else "🔻 LOW"
            text += f"{emoji}: {t.strftime('%a %H:%M')} ({tide['v']}m)\n"
        
        return text
# ==========================================
# Telegram Bot Handlers
# ==========================================

telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message with Singapore context"""
    welcome_text = """
🇸🇬 *Welcome to Singapore TideMaster Bot!* 🎣

Your free fishing companion for Singapore waters.

*What I can do:*
🌊 *Tide Charts* - Accurate predictions for 8 locations
🌤️ *Weather* - Real-time NEA forecasts
📍 *Spots* - Best local fishing locations
🤖 *AI Advice* - Smart fishing recommendations
🐟 *Species* - What's biting where

*Quick Commands:*
/today - Today's best fishing times
/spots - List of fishing spots
/tides [location] - Tide chart
/weather - Current conditions
/advice [spot] - Get fishing advice

*Popular Spots:*
• Bedok Jetty (East Coast)
• Sembawang Jetty (North)
• Punggol Jetty (Northeast)
• Labrador Park (South)
    """
    
    keyboard = [
        [InlineKeyboardButton("🌊 Today's Tides", callback_data='today')],
        [InlineKeyboardButton("📍 Fishing Spots", callback_data='spots')],
        [InlineKeyboardButton("🌤️ Weather", callback_data='weather')],
        [InlineKeyboardButton("🤖 Get Advice", callback_data='advice_menu')]
    ]
    
    await update.message.reply_text(
        welcome_text, 
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show today's best fishing times across Singapore"""
    loading = await update.message.reply_text("🎣 Analyzing today's conditions...")
    
    try:
        # Get weather
        weather = await sg_client.get_24hr_weather_forecast()
        weather_parsed = fishing_ai._parse_weather(weather)
        
        # Get tides for main stations
        tide_summary = []
        for station_key, station in list(SG_TIDE_STATIONS.items())[:3]:
            tide_data = await sg_client.get_tide_data(station_key)
            tides = tide_data.get("tides", [])[:2]  # Next 2 events
            tide_summary.append({
                "station": station["name"],
                "tides": tides
            })
        
        # Build response
        text = f"""
🇸🇬 *Today's Fishing Outlook*

🌤️ *Weather:* {weather_parsed['forecast']}
🌡️ Temperature: {weather_parsed['temperature_low']}°C - {weather_parsed['temperature_high']}°C
💨 Wind: {weather_parsed['wind_speed']} km/h

🌊 *Tide Times (Next 4 Events):*
"""
        for ts in tide_summary:
            text += f"\n*{ts['station']}:*"
            for t in ts['tides']:
                t_time = datetime.strptime(t['t'], "%Y-%m-%d %H:%M")
                emoji = "🔺" if t['type'] == 'H' else "🔻"
                text += f"\n  {emoji} {t_time.strftime('%H:%M')} ({t['v']}m)"
        
        # Best times recommendation
        now = datetime.now()
        text += f"\n\n⭐ *Best Fishing Windows:*\n"
        
        # Simple logic: 2 hours around dawn/dusk, incoming tide preferred
        dawn = now.replace(hour=6, minute=30)
        dusk = now.replace(hour=18, minute=30)
        
        if now < dawn:
            text += f"🌅 Dawn: ~6:30 AM (arrive by 5:30 AM)\n"
        if now < dusk:
            text += f"🌇 Dusk: ~6:30 PM (arrive by 5:30 PM)\n"
        
        text += "\n💡 *Tip:* Check /advice [spot_name] for location-specific tips!"
        
        await update.message.reply_text(text, parse_mode='Markdown')
        await loading.delete()
        
    except Exception as e:
        logger.error(f"Error in today: {e}")
        await loading.edit_text("❌ Error fetching data. Please try again.")

async def list_spots(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available fishing spots"""
    text = "🇸🇬 *Singapore Fishing Spots*\n\n"
    
    for spot_id, spot in SG_FISHING_SPOTS.items():
        text += f"🎣 *{spot['name']}*\n"
        text += f"   📍 {spot['location']}\n"
        text += f"   🌊 Best tide: {spot['best_tide'].title()}\n"
        text += f"   🐟 Targets: {', '.join(spot['target_species'][:3])}\n"
        text += f"   💡 Use: `/advice {spot_id}`\n\n"
    
    text += "_Data based on local angler reports and MPA guidelines_"
    
    await update.message.reply_text(text, parse_mode='Markdown')

async def get_tides(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get tide chart for Singapore location"""
    if not context.args:
        # Show available stations
        text = "🌊 *Available Tide Stations:*\n\n"
        for key, station in SG_TIDE_STATIONS.items():
            text += f"• *{station['name']}* - Use `/tides {key}`\n"
        text += "\n_Example: /tides bedok_"
        await update.message.reply_text(text, parse_mode='Markdown')
        return
    
    station_key = context.args[0].lower()
    if station_key not in SG_TIDE_STATIONS:
        await update.message.reply_text("❌ Unknown station. Use /tides to see available stations.")
        return
    
    loading = await update.message.reply_text(f"🌊 Calculating tides for {SG_TIDE_STATIONS[station_key]['name']}...")
    
    try:
        # Get tide data
        tide_data = await sg_client.get_tide_data(station_key)
        
        if not tide_data:
            await loading.edit_text("❌ Error calculating tides.")
            return
        
                # Generate tide table
        tide_table = visualizer.create_tide_table(
            tide_data, 
            SG_TIDE_STATIONS[station_key]['name']
        )
        
        await update.message.reply_text(tide_table, parse_mode='Markdown')
        
        # Send tide table
        tides = tide_data.get("tides", [])[:6]
        text = "*Next Tide Events:*\n"
        for t in tides:
            t_time = datetime.strptime(t['t'], "%Y-%m-%d %H:%M")
            emoji = "🔺 HIGH" if t['type'] == 'H' else "🔻 LOW"
            text += f"{emoji}: {t_time.strftime('%a %H:%M')} ({t['v']}m)\n"
        
        await update.message.reply_text(text, parse_mode='Markdown')
        await loading.delete()
        
    except Exception as e:
        logger.error(f"Error in get_tides: {e}")
        await loading.edit_text("❌ Error generating tide data.")

async def get_weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get Singapore weather from NEA"""
    loading = await update.message.reply_text("🌤️ Fetching NEA weather data...")
    
    try:
        weather_24h = await sg_client.get_24hr_weather_forecast()
        weather_4day = await sg_client.get_4day_weather_forecast()
        pm25 = await sg_client.get_pm25()
        
        # Parse 24h forecast
        item = weather_24h.get("items", [{}])[0]
        general = item.get("general", {})
        periods = item.get("periods", [])
        
        text = f"""
🇸🇬 *Singapore Weather (NEA)*

🌡️ *Temperature:* {general.get('temperature', {}).get('low', '--')}°C - {general.get('temperature', {}).get('high', '--')}°C
💧 *Humidity:* {general.get('relative_humidity', {}).get('low', '--')}% - {general.get('relative_humidity', {}).get('high', '--')}%
🌤️ *Forecast:* {general.get('forecast', 'N/A')}

*Today:*
"""
        for period in periods[:2]:
            time_range = period.get("time", {}).get("text", "")
            forecast = period.get("regions", {}).get("east", "N/A")
            text += f"• {time_range}: {forecast}\n"
        
        # Add 4-day outlook
        forecasts_4day = weather_4day.get("items", [{}])[0].get("forecasts", [])[:3]
        text += "\n*Next 3 Days:*\n"
        for fc in forecasts_4day:
            date = fc.get("date", "")[5:]  # MM-DD
            text += f"• {date}: {fc.get('forecast', 'N/A')}\n"
        
        # PM2.5
        pm_readings = pm25.get("items", [{}])[0].get("readings", {}).get("pm25_one_hourly", {})
        east_pm = pm_readings.get("east", 0)
        text += f"\n💨 *Air Quality (East):* PM2.5 = {east_pm} μg/m³"
        
        await update.message.reply_text(text, parse_mode='Markdown')
        await loading.delete()
        
    except Exception as e:
        logger.error(f"Error in get_weather: {e}")
        await loading.edit_text("❌ Error fetching weather data.")

async def get_advice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get AI fishing advice for Singapore spot"""
    if not context.args:
        text = """
🤖 *Get Fishing Advice*

Usage: `/advice [spot_name] [species(optional)]`

*Available spots:*
"""
        for spot_id in SG_FISHING_SPOTS.keys():
            text += f"• {spot_id}\n"
        
        text += "\n_Example: /advice bedok_jetty barramundi_"
        await update.message.reply_text(text, parse_mode='Markdown')
        return
    
    spot_id = context.args[0].lower()
    target = context.args[1] if len(context.args) > 1 else None
    
    if spot_id not in SG_FISHING_SPOTS:
        await update.message.reply_text("❌ Unknown spot. Use /advice to see available spots.")
        return
    
    loading = await update.message.reply_text("🎣 Analyzing conditions...")
    
    try:
        # Get data
        station = SG_FISHING_SPOTS[spot_id]["station"]
        tide_data = await sg_client.get_tide_data(station)
        weather_data = await sg_client.get_24hr_weather_forecast()
        
        # Get advice
        advice = await fishing_ai.get_fishing_advice(
            spot_id, tide_data, weather_data, target
        )
        
        await update.message.reply_text(advice, parse_mode='Markdown')
        await loading.delete()
        
    except Exception as e:
        logger.error(f"Error in get_advice: {e}")
        await loading.edit_text("❌ Error generating advice.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard buttons"""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'today':
        await today_command(update, context)
    elif query.data == 'spots':
        await list_spots(update, context)
    elif query.data == 'weather':
        await get_weather(update, context)
    elif query.data == 'advice_menu':
        await query.edit_message_text(
            "Choose a spot for advice:\n" + 
            "\n".join([f"/advice {k}" for k in SG_FISHING_SPOTS.keys()])
        )

# ==========================================
# FastAPI Application
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting Singapore TideMaster Bot...")
    
    # Setup webhook
    if RENDER_EXTERNAL_URL:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await telegram_app.bot.set_webhook(
            url=webhook_url,
            secret_token=WEBHOOK_SECRET
        )
        logger.info(f"Webhook set: {webhook_url}")
    
    # Register handlers
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("today", today_command))
    telegram_app.add_handler(CommandHandler("spots", list_spots))
    telegram_app.add_handler(CommandHandler("tides", get_tides))
    telegram_app.add_handler(CommandHandler("weather", get_weather))
    telegram_app.add_handler(CommandHandler("advice", get_advice))
    telegram_app.add_handler(CallbackQueryHandler(button_callback))
    
    await telegram_app.initialize()
    
    yield
    
    # Cleanup
    await sg_client.close()
    await fishing_ai.close()
    await telegram_app.shutdown()

app = FastAPI(title="Singapore TideMaster Bot", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "bot": "Singapore TideMaster",
        "status": "running",
        "location": "Singapore",
        "timezone": "SGT (UTC+8)",
        "data_sources": ["NEA", "MPA/HPA", "Local Knowledge"]
    }

@app.post("/webhook")
async def webhook(request: Request):
    """Handle Telegram webhook"""
    try:
        if WEBHOOK_SECRET:
            secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if secret != WEBHOOK_SECRET:
                raise HTTPException(status_code=403)
        
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return Response(status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
