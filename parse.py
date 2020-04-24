from __future__ import print_function
from html.parser import HTMLParser
import torch
import codecs
from bs4 import BeautifulSoup
import re


file = codecs.open("./transcripts/season01/0101.html", "r", "utf-8")

soup = BeautifulSoup(file.read(), 'html.parser')

reslist = []
current_scene_loc = ""

for line in soup.findAll('p'):
    text = line.get_text()
    text = re.sub('[^a-zA-Z0-9 \n\.:\'\[\]\,]', '', text)
    text = re.sub('\n', ' ', text)
    # reslist.append(text)
    locationline = re.search('\[Scene: (.*),(.*)\]', str(text))
    if locationline:
        current_scene_loc = locationline.group(1)
    else:
        voiceline = re.search('(.*): (.*)', str(text))
        if voiceline:
          person = voiceline.group(1)
          text = voiceline.group(2)
          reslist.append((current_scene_loc, person, text))
        
          


print(reslist)
