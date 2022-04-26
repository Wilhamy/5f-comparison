outfile = 'deleteme.out'

f = open(outfile, 'w')

f.write("Col1, Col2, Col3")
f.write(",Col4\n")

for i in range(12):
    f.write(f"{i},{2*i},{3*i},{4*i}")
    f.write('\n')

f.close()