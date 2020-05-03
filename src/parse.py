from bs4 import BeautifulSoup
import re
import os
import csv
import codecs
import textdistance


script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
input_dir = os.path.join(directory_path, "data/transcripts")  # path to transcripts dir
output_dir = os.path.join(directory_path, "data/parsed_transcripts")

def locations_same(a, b):
    return textdistance.levenshtein.normalized_similarity(a, b) > 0.7

allowed_locations = ['central perk', 'monica and rachels']


for season_name in os.listdir(input_dir):
    season_path = os.path.join(input_dir, season_name)  # path to season dir
    season_output_path = os.path.join(output_dir, season_name)
    if not os.path.exists(season_output_path):
        os.mkdir(season_output_path)

    for filename in os.listdir(season_path):  # for each episode in season
        episode_path = os.path.join(season_path, filename)  # path to episode
        episode_output_path =  os.path.join(season_output_path, os.path.splitext(filename)[0] + '.csv')

        episode_file = file = codecs.open(episode_path, "r", "latin-1")

        soup = BeautifulSoup(episode_file.read(), 'html.parser')
        text = soup.getText()
        text = text.splitlines()

        with open(episode_output_path, 'w', newline='') as csvfile:
            file_writer = csv.writer(csvfile, delimiter='|')
            file_writer.writerow(["location", "character", "line"])
            location = ''
            character = ''
            scentence = ''
            

            for line in text:
                line = re.sub('\n', ' ', line).lower()
                line = re.sub('[^a-zA-Z0-9 \n\.:\[\]\,;]', '', line)
                location_line = re.search('\[scene[:,.;] (.*?)[,.;\]](.*)', line)
                if location_line:
                    location = location_line.group(1)
                elif location:
                    voice_line = re.search('(.*?): (.*)', line)
                    if voice_line:
                        if character and scentence:
                            for loc in allowed_locations:
                                if locations_same(loc, location):
                                    csv_line = [loc, character, scentence]
                                    file_writer.writerow(csv_line)
                                    break
                            character, scentence = "", ""
                        character = voice_line.group(1)
                        scentence = voice_line.group(2)
                    elif(line == 'end'):
                        break
                    elif(re.match('[^\[\]]+$', line)):
                        scentence += " " + line
                    
                        
                            







