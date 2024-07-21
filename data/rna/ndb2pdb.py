ndb2pdb = {l[0].lower(): l[1].lower() for l in
           [l.split() for l in open('ndb_pdb_mapping.tsv', 'r').read().split('\n')[1:]] if len(l) == 2}

ndb = open('ndb_list.txt', 'r').read().split()


ndb_only = list()
pdb = list()
for n in ndb:
    try:
        print(ndb2pdb[n])
        pdb.append(ndb2pdb[n])
    except:
        ndb_only.append(n)

with open('list_file.txt', 'w') as out:
    out.write(','.join(pdb))