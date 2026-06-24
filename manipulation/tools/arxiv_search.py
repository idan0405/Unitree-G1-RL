"""Minimal arXiv API query helper used during research.

Prints the title and abstract of the top matches for a search query. Edit
`query` below to look up related work (default: RL + dexterous grasping).
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

query = 'all:"reinforcement learning" AND all:dexterous AND all:grasping'
url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&start=0&max_results=3&sortBy=relevance&sortOrder=descending'

try:
    response = urllib.request.urlopen(url)
    xml_data = response.read()
    root = ET.fromstring(xml_data)
    
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    
    for entry in root.findall('atom:entry', namespace):
        title = entry.find('atom:title', namespace).text.strip()
        summary = entry.find('atom:summary', namespace).text.strip()
        print("="*80)
        print("TITLE:", title)
        print("SUMMARY:\n", summary)
        print("="*80)
        
except Exception as e:
    print(f"Error: {e}")
