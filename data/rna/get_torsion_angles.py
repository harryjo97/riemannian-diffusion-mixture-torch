# Please see accompanying webpage:
#
# http://www.warwick.ac.uk/go/peter_cock/python/ramachandran/calculate/
#
# This code relies on Thomas Hamelryck's Bio.PDB module in BioPython:
#
# http://www.biopython.org


import math
import gzip
import shutil
import Bio.PDB
import os


def degrees(rad_angle):
    """Converts any angle in radians to degrees.

    If the input is None, the it returns None.
    For numerical input, the output is mapped to [-180,180]
    """
    if rad_angle is None:
        return None
    angle = rad_angle * 180 / math.pi
    while angle > 180:
        angle = angle - 360
    while angle < -180:
        angle = angle + 360
    return angle


# noinspection PyShadowingNames
def base_type(residue, next_residue):
    """Expects Bio.PDB residues, returns ramachandran 'type'

    If this is the last residue in a polypeptide, use None
    for next_residue.

    Return value is a string: "General", "Glycine", "Proline"
    or "Pre-Pro".
    """
    if residue.resname.upper() == "GLY":
        return "Glycine"
    elif residue.resname.upper() == "PRO":
        return "Proline"
    elif next_residue is not None and next_residue.resname.upper() == "PRO":
        # exlcudes those that are Pro or Gly
        return "Pre-Pro"
    else:
        return "General"


pdb_code = "437d"
output_file = open("aggregated_angles.tsv", "w")
for pdb_code in open('list_file.txt', 'r').read().split(','):

    filename = pdb_code + '.pdb'
    if not os.path.isfile(pdb_code):
        gzname = filename + '.gz'
        with gzip.open(gzname, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print("About to load Bio.PDB and the PDB file...")
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, "%s.pdb" % pdb_code)
    print("Done")

    print("About to save angles to file...")
    # output_file = open("%s_biopython.tsv" % pdb_code,"w")

    consecutive_residuals = [[], []]
    for model in structure:
        for chain in model:
            print("Chain %s" % str(chain.id))
            for r in chain:
                resname = r.resname.replace(' ', '')
                if resname in ['A', 'C', 'G', 'U']:
                    print(r)

                    consecutive_residuals[-2].append(r)
                    consecutive_residuals[-1].append(r)
                    consecutive_residuals.append([r])

                else:
                    consecutive_residuals.extend([[], []])

    consecutive_residuals = [rs for rs in consecutive_residuals if len(rs) == 3]

    for rs in consecutive_residuals:
        try:
            # backbone
            r = rs[1]
            O3_prev = Bio.PDB.Vector(rs[0]["O3'"].coord)
            P = Bio.PDB.Vector(r["P"].coord)
            O5_ = Bio.PDB.Vector(r["O5'"].coord)
            C5_ = Bio.PDB.Vector(r["C5'"].coord)
            C4_ = Bio.PDB.Vector(r["C4'"].coord)
            C3_ = Bio.PDB.Vector(r["C3'"].coord)
            O3_ = Bio.PDB.Vector(r["O3'"].coord)

            P_next = Bio.PDB.Vector(rs[1]["P"].coord)
            O5_next = Bio.PDB.Vector(rs[1]["O5'"].coord)

            alpha = Bio.PDB.calc_dihedral(O3_prev, P, O5_, C5_)
            beta = Bio.PDB.calc_dihedral(P, O5_, C5_, C4_)
            gamma = Bio.PDB.calc_dihedral(O5_, C5_, C4_, C3_)
            delta = Bio.PDB.calc_dihedral(C5_, C4_, C3_, O3_)
            epsilon = Bio.PDB.calc_dihedral(C4_, C3_, O3_, P_next)
            zeta = Bio.PDB.calc_dihedral(C3_, O3_, P_next, O5_next)

            resname = r.resname.replace(' ', '')
            if resname in ['C', 'U']:
                chi = Bio.PDB.calc_dihedral(
                    Bio.PDB.Vector(r["O4'"].coord),
                    Bio.PDB.Vector(r["C1'"].coord),
                    Bio.PDB.Vector(r["N1"].coord),
                    Bio.PDB.Vector(r["C2"].coord))
            else:
                chi = Bio.PDB.calc_dihedral(
                    Bio.PDB.Vector(r["O4'"].coord),
                    Bio.PDB.Vector(r["C1'"].coord),
                    Bio.PDB.Vector(r["N9"].coord),
                    Bio.PDB.Vector(r["C4"].coord))

            angles = [alpha, beta, gamma, delta, epsilon, zeta, chi]
            angles = [degrees(a) for a in angles]

            output_file.write("%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"
                              % (pdb_code, resname,
                                 angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], angles[6]))
        except:
            print(f'{pdb_code}')

output_file.close()
print("Done")


