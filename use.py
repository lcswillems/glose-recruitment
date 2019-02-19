import argparse
import os
import urllib.parse

from utils import GloseEntity, find_wikipedia_page_url, get_model_path
from preprocess import preprocess_text, id_to_label

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
parser.add_argument("--model", default=None,
                    help="Model name to recognize named entities.")
parser.add_argument("--use-spacy", action="store_true", default=False,
                    help="Use spaCy library to recognize named entities.")
args = parser.parse_args()

assert (args.i != None) ^ (args.i_fname != None), "Exactly one input type must be defined."
assert (args.model != None) ^ (args.use_spacy), "spaCy or a trained model must be used."

# Load input

if args.i:
    text = args.i
    i_bname = "txt"
elif args.i_fname:
    with open(args.i_fname) as f:
        text = f.read()
    i_bname = os.path.splitext(args.i_fname)[0]

# Recognize named entities

if args.use_spacy:
    import spacy
    nlp = spacy.load("en_core_web_sm")

    ents = nlp(text).ents
    glose_ents = []
    for ent in ents:
        if ent.label_ == "PERSON":
            cat = "PER"
        elif ent.label_ in ["ORG", "LOC"]:
            cat = ent.label_
        else:
            cat = "MISC"
        glose_ent = GloseEntity(ent.text, ent.start_char, ent.end_char, cat)
        glose_ents.append(glose_ent)
elif args.model != None:
    from keras.models import load_model

    lword_id_sents, casing_id_sents, pos_sents = preprocess_text(text)

    model_path = get_model_path(args.model)
    model = load_model(model_path)
    pred_label_id_sents = model.predict(casing_id_sents).argmax(axis=2)

    # Create entities from model predictions

    glose_ents = []

    for pred_label_id_sent, pos_sent in zip(pred_label_id_sents, pos_sents):
        # Remove padding
        pred_label_id_sent = pred_label_id_sent[-len(pos_sent):]

        ent = None

        def end_entity():
            global ent
            if ent != None:
                ent_text = text[ent["start"]:ent["end"]]
                glose_ent = GloseEntity(ent_text, ent["start"], ent["end"], ent["cat"])
                glose_ents.append(glose_ent)
                ent = None

        for pred_label_id, pos in zip(pred_label_id_sent, pos_sent):
            l = id_to_label[pred_label_id]
            l_prefix, l_cat = (l.split("-")+[""])[:2] # Add [""] for case l == "O"
            if l_prefix == "O":
                end_entity()
            elif l_prefix == "B":
                end_entity()
                ent = {"start": pos[0], "end": pos[1], "prefix": l_prefix, "cat": l_cat}
            elif l_prefix == "I":
                if ent != None and l_cat == ent["cat"]:
                    ent["end"] = pos[1]
                else:
                    end_entity()

# Add Wikipedia links to non-misc entities if asked

if args.wikipedia_link:
    for ent in glose_ents:
        ent.wikipedia_url = None
        if ent.cat != "MISC":
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

text_pos = 0
for ent in glose_ents:
    ent_text = ent.text
    if hasattr(ent, "wikipedia_url") and ent.wikipedia_url != None:
        ent_text = '<a href="{}">{}</a>'.format(ent.wikipedia_url, ent_text)

    output += text[text_pos:ent.start]
    output += '<span class="{}">{}</span>'.format(ent.cat, ent_text)

    text_pos = ent.end
output += text[text_pos:]

output += "</div>"

# Save output

if args.o_fname != None:
    o_fname = args.o_fname
else:
    recognizer = "spaCy" if args.use_spacy else "model " + args.model
    o_fname = "{}_with_links (by {}).html".format(i_bname, recognizer)

with open(o_fname, "w") as f:
    f.write(output)

# Display helper to open the file

o_fname_url = "file://" + urllib.parse.quote(os.path.abspath(o_fname))
print("Output filename url:\n" + o_fname_url)