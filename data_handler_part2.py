import csv
import numpy as np
import os

dir = os.path.dirname(__file__)


outfilecode = 0
outfile = None

for num in range(2399):                            ## range(2399)
    print('num is ' + str(num))
    with open('master_out.csv', 'r') as master:
        reader = csv.reader(master, delimiter=',')
        for row in reader:
            cnum = int(row[0])
            if cnum == num:
                if outfile is not None:
                    outfile.close()
                outfilename = os.path.expanduser('~/Dropbox/Documents/Python/csv_tests/output/{}.csv'.format(num))
                outfile = open(outfilename, 'a')
                current_text = row[1]
                outfile.write(current_text + '\n')
                print(num)
                print(row[1])
                outfile.close()