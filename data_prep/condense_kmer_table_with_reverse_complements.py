import sys
from collections import defaultdict
from inference import tools

def main(argv):
    if len(argv) != 2:
        sys.stderr.write("python"
                " condense_kmer_table_with_reverse_complements.py"
                " kmer_table.tsv\n")
        sys.exit(2)
    kmertable = defaultdict(int)
    kmer_table_f = open(argv[1])
    for l in kmer_table_f:
        kmer, count = l.strip("\n").split("\t")
        count = int(count)
        if kmertable.has_key(tools.rev_comp(kmer)):
            kmertable[tools.rev_comp(kmer)]+=count
        else:
            kmertable[kmer]+=count
    kmer_table_f.close()
    for (kmer, count) in kmertable.iteritems():
        print "%s\t%d"%(kmer,count)

if __name__ == "__main__":
    main(sys.argv) 
