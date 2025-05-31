original = "./joints2smpl/smpl_models/gmm_08.pkl"
destination = "./joints2smpl/smpl_models/dos_gmm_08.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    
    content = infile.read()
with open(destination, 'wb') as output:
    
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))

