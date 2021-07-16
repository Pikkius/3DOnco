# 3DOnco
Model for Onco prediction using 3D protein structure

# Table of Contents

- [Data](#Data)
- [Gene Fusion](#Gene)
- [Protein Structure Prediction](#psp)
  - [HHBlits](#hhblits)
  - [DCA](#dca)
  - [ProSPr](#prospr)
- [Models](#models)
- [Results](#results)


---

# Data <a name="Data"></a>
BLA BLA
BLA
BLA
BLA 
NON SO 

# Gene Fusion <a name="Gene"></a>

# Theory
Gene fusions are specific kind of aberrations that happen when parts of two different genes join together as result of a translocation, interstitial deletion or chromosomal inversion. Fusion proteins resulting from the expression of these hybrid genes may lead to the development of different pathologies, especially cancers: in this case the gene under analysis is defined as 'Oncogene'.

In this scenario, the coordinates of the base pair at which the 2 genes are fused together is called breakpoint, so we refer to the gene BEFORE the break point as <b>5' gene </b> and to the gene AFTER the break point as <b>3' gene </b>.

An eukaryotic gene is characterised by different areas:

https://www.researchgate.net/figure/An-illustration-of-a-typical-structure-of-a-eukaryotic-gene-A-gene-may-have-many_fig3_51510353 

the ones of interest for our study are the coding DNA sequences (CDS) which are the regions that are transcribed into RNA and translated into proteins. Moreover, a single gene can produce multiple different RNAs due to splicig procedure (?) that are called _transcripts_ and, for each gene, we consider one single transcript that is the longest one; also, if two transcripts have the same length, than we consider the one with the highest number of coding sequences. 

Building gene fusions sequences requires to consider two important things:
* if we are dealing with the 5' gene (first gene of the fusion) or the 3' gene (second gene of the fusion)
* it the gene transcribes in the + or in the - strand. In this case the - strand must be reversed and the bases must be substituted with their complementaries, since the databases contain only + strand genes.

With 2 genes and 2 signs this leads to 4 different cases: 
IMMA


If we consider the 5' gene and it transcribes in the + strand, or the 3' gene that transcribes in the - strand, the portion of the gene that preceeds the breakpoint is selected. (FAcciamo uno schemino??)

On the other hand, if we consier the 5' gene that transcribes in the - strand or the 5+ gene that transcribes in the + strand, we take the portion of the gene that follows the breakpoint.  (Anche qui??)

Moreover we have to take into account the position of the break points. 

If the break point is inside a CDS for both the genes the transcripts are merged together based on the previous rules.

But we need to be more carefull if the break point is inside an exon or inside an Untranslated Region (UTR):
SCHEMA

5' gene transcribes with + sign 

bp into an exon we should stop the transcription of this gene at the preceding CDS.

5' UTR --> NULL
3' UTR --> GENE COMPLETE

5' gene transcribes with - sign 

bp into an exon we should stop the transcription of this gene at the following CDS. (perchè stiamo sempre leggendo sul + strand)

5' UTR --> GENE COMPLETE
3' UTR --> NULL

3' gene transcribes with + sign 

bp into an exon we should start the transcription of this gene at the following CDS.

5' UTR --> 
3' UTR --> 

3' gene transcribes with - sign 

bp into an exon we should start the transcription of this gene at the preceding CDS. (perchè stiamo sempre leggendo sul + strand)

5' UTR --> 
3' UTR --> 

FINE SCHEMA

#FUNCTION
(Implemented on Colab gli diamo qualche specifica?)
#FILTER GENOME

Firstly we need to filter the human genome, to do so we implemented Gene_Fusion.ipyn that takes in input a gtf file which contains all the genome annotated. The algorithm filter the data considering the ensembl notation and gives as result a correspondence one to one between a gene and its transcript. Since one gene can be associated to more tha one transcript due to splicing mechanims in this case we selected the transcript that contains more CDS.
At the end the results is ordered by chromosomes and stored in a csv.
The ipyn takes as input argument the version of the annotation, for semplicity we had already runned the file using GRCH37 and GRCH38. The csv results are inside the folder (NON LO SO), so you don't have to run again this part. For future updatings the time of run is about 20 minutues (di più). GRCH files can be downloaded at LINK

ESEMPIO PER GLI SCEMI ---> ADELA

#GENE FUSION

We prepared a function to simulate gene fusion in Gene_fusion_Andre.ipyn.
The script takes as input 2 chromosome and 2 break points and then simulate the gene fusion. The first pair is associated to the 5' gene and the second one to the 3' gene.
Moreover, the code to run need the gtf files for all the 48 chromosome and the csv file generated in the previous point. You can download the files from LINKS

SCRIVO COME FUNZIONA (?)

The function gives as ouput a fasta file that contains the protein generated by the fusion. Moreover in the 

( Another important thing to takes into account is that the final sequences of genes that transcribe in the - strand must be reversed and the bases must be substituted with their complementaries. lo metterei su )

Our model .... spiega modello 
input 
output 


# Protein Structure Prediction <a name="psp"></a>

Protein structure prediction is one of the most interesting tasks in bioinformatics field. The structure of the protein can be analyzed and considered from four different levels: primary, secondary, tertiary and quaternary. The _primary protein structure_ consists of the sequence of amino acids, that are the monomers that constitute proteins; the _secondary structure_ is the local folding of the polypeptide chain into alpha helices and beta sheets elements; the _tertiary structure_ describes the 3D shape of the protein molecule, is composed by a single polypeptide chain backbone with one or more secondary structures and reveals very important functional and chemical properties of the protein; the _quaternary structure_ is the association of several protein chains. 

In this context, we make use of a folding algorithm that exploits deep neural networks techniques to predict the protein tertiary structure from its amino acid sequence.

## HHBlits <a name="hhblits"></a>

We start from the amino acid sequences extracted from the FASTA files obtained as a result of the gene fusion step. The second step is to generate alignments using the tool HHBlits from HHSuite.

One of the main processes in computational biology consists of building protein **multiple-sequence alignments** (MSAs) since MSAs are a key intermediate step in the prediction of protein tertiary structure. MSA refers to a sequence alignment of more than two sequences from which sequence homology can be inferred. 

HHBlits is one of the most important tools to generate multiple sequence alignments. The starting point is to convert the sequences to a condensed representation of a MSA called **profile hidden Markov model** (HMM). HMMs are similar to sequence profiles, in the sense that they transform MSA into _position-specific scoring systems_: profile-profile comparison methods are preferred to sequence-sequence comparison since profiles give much more information than the simple sequence and they are more powerful.  In sequence profiles, each of the 20 amino acids is assigned with a score for each position in the query sequence: this score corresponds to its frequency in that position in the original alignment. These frequencies can be interpreted as probabilities of seen that amino acid in new related proteins. Profile HMM provide, in addition to the amino acid frequencies, also information about the frequencies of insertions and deletions at each column.

After creating the HMM, the HHBlits server iteratively searches through an HMM database - in our case, UniProt30: it looks for the sequences with an expected value (_E_ value) below a certain threshold and it adds them to the query MSA, from which the HMM for the next search iteration is built. 


HHBlits is a very fast and sensitive algorithm thanks to a two-stage prefilter phase that reduces the number of database HMMs to be aligned. 

## Direct-Coupling Analysis <a name="dca"></a>
The alignment results produced with HHBlits are then used to fit a statistical model called Direct-Coupling Analysis (DCA). 

## ProSPr <a name="prospr"></a>

# Models <a name="models"></a>

# Results <a name="results"></a>
