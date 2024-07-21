`batch_download.sh` is taken from 
from https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script
* used to download the list of pdb files in `list_file.txt`.
* `list_file.txt` obtained from 
https://warwick.ac.uk/fac/sci/moac/people/students/peter_cock/python/ramachandran/top500

In this folder, 
* run `./batch_download.sh -f list_file.txt -p`
* run `python get_torsion_angle.py` to read all pdb files and aggregate all angle data in `aggregated_angles.tsv`
