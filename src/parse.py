import os
import re

import pandas as pd
import textdistance
from bs4 import BeautifulSoup

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(
    script_path)[0])  # path to python script dir
# path to transcripts dir
input_dir = os.path.join(directory_path, "data/transcripts")

ALLOWED_LOCATIONS = ['central perk', 'monica and rachels']

CURRENT_LOCATION = ''


def locations_same(a, b):
    return textdistance.levenshtein.normalized_similarity(a, b) > 0.7


def allowed_location(location):
    if location == '':
        return False

    for loc in ALLOWED_LOCATIONS:
        if locations_same(loc, location):
            return loc
    return False


def parse_line(line):
    global CURRENT_LOCATION

    soup = BeautifulSoup(line, 'html.parser')
    line = soup.get_text().lower()

    location_line = re.search('\[scene[:,.;] (.*?)[,.;\]](.*)', line)
    if location_line:
        CURRENT_LOCATION = location_line.group(1)
        return None

    voice_line = re.search('(.*?): (.*)', line)
    if voice_line and CURRENT_LOCATION:
        character = voice_line.group(1)
        scentence = voice_line.group(2)
        return [character, scentence]


def parse():
    global CURRENT_LOCATION
    data = []
    for season_name in os.listdir(input_dir):
        season_path = os.path.join(
            input_dir, season_name)  # path to season dir

        for filename in os.listdir(season_path):  # for each episode in season
            episode_path = os.path.join(
                season_path, filename)  # path to episode
            with open(episode_path, 'r', encoding='latin-1') as episode_file:
                current_line = ''
                for line in episode_file:
                    if line == '\n':
                        parsed = parse_line(current_line)
                        current_line = ''
                        parsed_loc = allowed_location(CURRENT_LOCATION)
                        if parsed and parsed_loc:
                            data.append(
                                [filename, parsed_loc, parsed[0], parsed[1], CURRENT_LOCATION])
                    else:
                        current_line += line

    data = pd.DataFrame(
        data, columns=['episode', 'location', 'character', 'line', 'original_location'])
    parsed_path = os.path.join(directory_path, "data/parsed_transcripts.csv")
    data.to_csv(parsed_path)


parse()
