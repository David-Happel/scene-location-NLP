from __future__ import print_function
from html.parser import HTMLParser
import torch
import codecs
from bs4 import BeautifulSoup
import re


file = codecs.open("./transcripts/season01/0101.html", "r", "utf-8")

soup = BeautifulSoup(file.read(), 'html.parser')
text = soup.getText()
text = text.splitlines()


reslist = []
current_scene_loc = ""

for line in text:
    line = re.sub('\n', ' ', line).lower()
    line = re.sub('[^a-zA-Z0-9 \n\.:\'\[\]\,]', '', line)
    locationline = re.search('\[scene: (.*?),(.*)\]', line)
    if locationline:
        current_scene_loc = locationline.group(1)
    elif current_scene_loc:
        voiceline = re.search('(.*?): (.*)', line)
        if voiceline:
            person = voiceline.group(1)
            t = voiceline.group(2)
            reslist.append((current_scene_loc, person, t))
        elif(line == 'end'):
            break
        else:
            reslist[-1] = (reslist[-1][0], reslist[-1][1], reslist[-1][2] + " " + line)
  

    
    
   
          


print(reslist)
