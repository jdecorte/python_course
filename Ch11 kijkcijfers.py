import datetime
import pytz
import time
import os
import sys
import requests
import json
import numpy as np
import pandas as pd 
import google.generativeai as genai
import openmeteo_requests
import requests_cache
from retry_requests import retry

scrape = False
genre_prediction = False
weather_history = True
debug = False

class Data:
    def __init__(self):
        self.records = [] # list of dictionaries

    def create_df(self):
        self.df = pd.DataFrame(self.records)
  
    def load_df(self):
        self.df = pd.read_csv(self.csv,sep=";")

    def save_df(self):
        self.df.to_csv(self.csv, index=False, sep=";")

    def my_name(self):
        if debug: # print the name of the calling function
            print (f"Calling {sys._getframe(1).f_code.co_name}")

class KijkCijfers(Data):
    def __init__(self, start_date, end_date):
        super().__init__()  
        self.start_date = start_date
        self.end_date = end_date
        self.url = "https://api.cim.be/api/cim_tv_public_results_daily_views?dateDiff={date}&reportType=north"
        self.csv = "kijkcijfers.csv"

    def scrape_data(self):
        current_date = self.start_date
        while current_date <= self.end_date:
            year = current_date.year
            date_str = current_date.strftime("%Y-%m-%d")
            current_url = self.url.format(date=date_str)
            print(f"Fetching data for {date_str}...")
            try:
                response = requests.get(current_url)
                if response.status_code == 200:
                    data = response.json()
                    data = data.get('hydra:member')
                    for item in data:
                        record = {
                            'ranking': item.get('ranking'),
                            'description': item.get('description').upper(),
                            'channel': item.get('channel').upper(),
                            'dateDiff': item.get('dateDiff'),
                            'startTime': item.get('startTime'),
                            'rLength': item.get('rLength'),
                            'rateInk': item.get('rateInK')
                        }
                        for key in record:
                            val = record[key]
                            if type(val) == str:
                                record[key] = val.replace('"', '')  # remove double quotes

                        if len(record['description']) > 0: # only keep records with a description
                            self.records.append(record)
                else:
                    print(f"Request failed with status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
            current_date += datetime.timedelta(days=1)

    def clean_data(self):
        return
    
    def convert_names(self):
        conversion_table = {
            "VIER": "PLAY4",
            "EEN": "VRT 1",
            "CANVAS": "VRT CANVAS",
            "Q2": "VTM2",
            "VITAYA": "VTM3",
            "CAZ": "VTM4",
            "ELEVEN PRO LEAGUE 1 NL":"DAZN PRO LEAGUE 1 (NL)"
        }
        convert = lambda x: conversion_table[x] if x in conversion_table else x
        self.df['channel'] = self.df['channel'].apply(convert) 

class Genre(Data):
    def __init__(self,kijkcijfers_df):
        super().__init__()
        self.my_name()
        api_key = open("api_key.txt").read().strip()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        # limits of free tier
        self.requests_per_minute = 15
        self.tokens_per_minute = 1000000  # not used
        self.requests_per_day = 1500 
        self.csv = "genres.csv"

        # self.records = [] # list of dictionaries
        self.kijkcijfers_df = kijkcijfers_df
        self.df = None
        self.count = 0
        self.buffer_size = 20

    def predict_buffer(self, buffer):
        self.my_name()
        idle_time = 60 / self.requests_per_minute + 1  # add 1 second to be sure
        self.count += len(buffer)
        text = "Geef voor volgende kanaal/programma-combinaties genre en subgenre (Nederlandstalig) in json-formaat met items kanaal, programma, genre en subgenre:"
        text += "(Bepaal genre en subgenre zodat gelijkaardige programma's hetzelfde genre en subgenre krijgen.)\\"     
        for combo in buffer:
            text += combo[0] + ":" + combo[1] + "\\"
        response = self.model.generate_content(text)
        data = response.text.replace('```json','').replace('```','')
        genre = json.loads(data)

        for item in genre:
            record = {
            'channel': item['kanaal'],
            'program': item['programma'],
            'genre': item['genre'],
            'subgenre': item['subgenre'],
            'aantal': 1
            }
            self.records.append(record)
        time.sleep(idle_time)
        return

    def predict_genres(self):
        self.my_name()
        # Create a set to track unique (channel, program) combinations
        unique_combinations = set()
        buffer = []
        for index, row in self.kijkcijfers_df.iterrows():
            try:
                channel = row['channel']
                program = row['description']
                combination = (channel, program)

                if combination not in unique_combinations:
                    print(f"Predicting genre for {channel} - {program}")
                    buffer.append(combination)
                    unique_combinations.add(combination)
                    if len(buffer) == self.buffer_size:
                        self.predict_buffer(buffer)
                        buffer = []
                else:
                    print(f"Skipping {channel} - {program}")
                    # Find the existing record and update 'aantal'
                    for record in self.records:
                        if record['channel'] == channel and record['program'] == program:
                            record['aantal'] += 1
                            break
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

    def convert_names(self):
        conversion_table = {
            "Actiefilm": "Actie",
            "Actua": "Actualiteit",
            "Actualiteiten": "Actualiteit",
            "Actueel": "Actualiteit",
            "Animatiefilm": "Animatie",
            "Avonturenfilm": "Avontuur",
            "Comedyserie": "Comedy",
            "Comedie": "Comedy",
            "Crimi": "Misdaad",
            "Crime": "Misdaad",
            "Misdaadserie": "Misdaad",
            "Familiefilm": "Familie",
            "Fantasiefilm": "Fantasie",
            "Fantasy": "Fantasie",            
            "Gespreks": "Gesprek",        
            "Gespreksprogramma": "Gesprek",
            "Gespreks-programma": "Gesprek",
            "Gesprekshow": "Gesprek",
            "Humor & Entertainment": "Humor",
            "Informatief": "Informatie",
            "Kinderprogramma": "Kinderen",
            "Kookprogramma": "Koken",
            "Natuurdocumentaire": "Documentaire",
            "Nieuws en Actualiteit": "Actualiteit",
            "Nieuws & Actualiteit": "Actualiteit",
            "Nieuws & Actueel": "Actualiteit",
            "Reality-tv": "Reality",
            "Reality-TV": "Reality",
            "Reisprogramma": "Reis",
            "Romance": "Romantiek",
            "Romantische Komedie": "Romantiek",
            "Romantische film": "Romantiek",
            "Talentenshow": "Talent",
            "Science Fiction": "Sciencefiction",
            "Sciencefictionfilm": "Sciencefiction",
            "Sci-fi": "Sciencefiction",
            "VariÃ©tÃ©" : "Variété"
            }
        convert = lambda x: conversion_table[x] if x in conversion_table else x
        self.df['genre'] = self.df['genre'].apply(convert)   

class Weather(Data):
    def __init__(self):
        super().__init__()
        # Setup the Open-Meteo API client with cache and retry on error
        self.cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        self.retry_session = retry(self.cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = self.retry_session)
        self.csv = "weather.csv"

    def get_weather(self,start_date, end_date):
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 50.8505,
            "longitude": 4.3488,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ["temperature_2m", "precipitation"],
            "timezone": "Europe/Berlin"
        }
        responses = self.openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["precipitation"] = hourly_precipitation

        self.df = pd.DataFrame(data = hourly_data)
        # print(self.df)

def main():
    start_date = datetime.date(2017, 1, 1)
    end_date = datetime.date(2025, 3, 26)

    kijkcijfers = KijkCijfers(start_date, end_date)
    print(kijkcijfers.start_date)
    if scrape:
        kijkcijfers.scrape_data()
        kijkcijfers.create_df()
        kijkcijfers.clean_data()
        kijkcijfers.save_df()
    else:  # postprocessing
        kijkcijfers.load_df()
        kijkcijfers.convert_names()
        kijkcijfers.save_df()

    if genre_prediction:
        genre = Genre(kijkcijfers.df)
        genre.predict_genres()
        print(f"Aantal programma's = {genre.count}") 
        genre.convert_names
        genre.create_df()
        genre.save_df()
    else: # postprocessing
        genre = Genre(kijkcijfers.df)
        genre.load_df()
        genre.convert_names()
        # genre.save_df()

    if weather_history:
        weather = Weather()
        weather.get_weather(start_date, end_date)
        weather.save_df()
   
if __name__ == "__main__":
    main()
