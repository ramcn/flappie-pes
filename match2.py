from Bio import pairwise2
from Bio.pairwise2 import format_alignment
with open('original.maf', 'r') as file:
  string1 = file.read().replace('\n', '')

with open('modified.maf', 'r') as file:
  string2 = file.read().replace('\n', '')

alignments = pairwise2.align.globalxx(string1, string2)
print(format_alignment(*alignments[0]))
print("Total string Length ", len(string1))
print("Match percentage ", len(string2)/len(string1)*100)
