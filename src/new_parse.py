import codecs
import os
import re

import numpy as np
import pandas as pd
import textdistance
from bs4 import BeautifulSoup

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
input_dir = os.path.join(directory_path, "data/transcripts")  # path to transcripts dir
output_path = os.path.join(directory_path, "data/parsed_transcripts.csv")

def locations_same(a, b):
    return textdistance.levenshtein.normalized_similarity(a, b) > 0.7

allowed_locations = ['central perk', 'monica and rachels']

data = []
for season_name in os.listdir(input_dir):
    season_path = os.path.join(input_dir, season_name)  # path to season dir

    for filename in os.listdir(season_path):  # for each episode in season
        episode_path = os.path.join(season_path, filename)  # path to episode
        episode_file = file = codecs.open(episode_path, "r", "latin-1")

        soup = BeautifulSoup(episode_file.read(), 'html.parser')
        text = soup.getText()
        text = text.splitlines()

        location = ''
        character = ''
        scentence = ''
        
        for line in text:
            line = re.sub('\n', ' ', line).lower()
            line = re.sub('[^a-zA-Z0-9 \n\.:\[\]\,;]', '', line)
            line = re.sub('\.\.+', '', line)
            location_line = re.search('\[scene[:,.;] (.*?)[,.;\]](.*)', line)
            if location_line:
                location = location_line.group(1)
            elif location:
                voice_line = re.search('(.*?): (.*)', line)
                if voice_line:
                    if character and scentence:
                        for loc in allowed_locations:
                            if locations_same(loc, location):
                                csv_line = [filename, loc, character, scentence]
                                data.append(csv_line)
                                break
                        character, scentence = "", ""
                    character = voice_line.group(1)
                    scentence = voice_line.group(2)
                elif(line == 'end'):
                    break
                elif(re.match('[^\[\]]+$', line)):
                    scentence += " " + line                       
                            
data = pd.DataFrame(data, columns=['episode','location', 'character', 'line'])
data["line"] = data["line"].str.split("\. ")
data = data.explode("line").sample(frac=1).reset_index(drop=True)
data = data.replace('', np.nan).dropna()
data.to_csv(output_path)






