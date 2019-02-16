import argparse
import os
import urllib.parse

from utils import GloseEntity, find_wikipedia_page_url

# Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--i", default=None,
                    help="Input text.")
parser.add_argument("--i-fname", default=None,
                    help="Filename containing the input text.")
parser.add_argument("--o-fname", default=None,
                    help="Filename that will contain the output. Default: [i_bname]_with_links.html, where [i_bname] equals --i-fname basename if --i-fname is defined otherwise 'txt'.")
parser.add_argument("--wikipedia-link", action="store_true", default=False,
                    help="Enrich output with links to Named Entities Wikipedia page.")
args = parser.parse_args()

assert (args.i != None) ^ (args.i_fname != None), "Exactly one input type must be defined."

# Load input

if args.i:
    txt = args.i
    i_bname = "txt"
elif args.i_fname:
    with open(args.i_fname) as f:
        txt = f.read()
    i_bname = os.path.splitext(args.i_fname)[0]

# Recognize named entities

## TODO:
## TODO: reconnaître seulement les bonnes catégories

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

ents = nlp(txt).ents
glose_ents = []
for ent in ents:
    if ent.label_ == "PERSON":
        label = "PER"
    elif ent.label_ in ["ORG", "LOC"]:
        label = ent.label_
    else:
        label = "MISC"
    glose_ent = GloseEntity(ent.text, ent.start_char, ent.end_char, label)
    glose_ents.append(glose_ent)

# Add Wikipedia links to non-misc entities if asked

if args.wikipedia_link:
    for ent in glose_ents:
        ent.wikipedia_url = None
        if ent.label != "MISC":
            ent.wikipedia_url = find_wikipedia_page_url(ent.text)

# Generate HTML output

output = """
<style>
body { line-height: 1.4em }
.PER { background-color: #ffda77 }
.ORG { background-color: #82ff92 }
.LOC { background-color: #82d5ff }
.MISC { background-color: #f187ff }
</style>
<div>
<span class="PER">PER</span>
<span class="ORG">ORG</span>
<span class="LOC">LOC</span>
<span class="MISC">MISC</span>
</div><br>
"""

output += "<div>"

txt_pos = 0
for ent in glose_ents:
    ent_text = ent.text
    if hasattr(ent, "wikipedia_url") and ent.wikipedia_url != None:
        ent_text = '<a href="{}">{}</a>'.format(ent.wikipedia_url, ent_text)

    output += txt[txt_pos:ent.start]
    output += '<span class="{}">{}</span>'.format(ent.label, ent_text)

    txt_pos = ent.end
output += txt[txt_pos:]

output += "</div>"

# Save output

if args.o_fname != None:
    o_fname = args.o_fname
else:
    o_fname = i_bname + "_with_links.html"

with open(o_fname, "w") as f:
    f.write(output)

# Display helper to open the file

o_fname_url = "file://" + urllib.parse.quote(os.path.abspath(o_fname))
print("Output filename url:\n" + o_fname_url)