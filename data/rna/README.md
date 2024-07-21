`batch_download.sh` is taken from 
from https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script
* used to download the list of pdb files in `list_file.txt`.
* `list_file.txt` obtained by running `python ndb2pdb.py`, which converts a list of ndb files into pdb tags, taken from
https://www.pnas.org/doi/epdf/10.1073/pnas.1835769100
(The original list has 132 files. We take 113 of them that also have pdb tags.)


In this folder, 
* run `./batch_download.sh -f list_file.txt -p`
* run `python get_torsion_angle.py` to read all pdb files and aggregate all angle data in `aggregated_angles.tsv`
