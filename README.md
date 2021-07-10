# 3DOnco
Model for Onco prediction using 3D protein structure

# Table of Contents

- [Data](#Data)
- [Gene Fusion](#Gene)
- [Protein Structure Prediction](#psp)
  - [HHBlits](#hhblits)
  - [ProSPr](#prospr)
- [Models](#models)
- [Results](#results)


---

# Data <a name="Data"></a>

# Gene Fusion <a name="Gene"></a>

# Protein Structure Prediction <a name="psp"></a>

Protein structure prediction is one of the most interesting tasks in bioinformatics field. The structure of the protein can be analyzed and considered from four different levels: primary, secondary, tertiary and quaternary. The _primary protein structure_ consists of the sequence of amino acids, that are the monomers that constitute proteins; the _secondary structure_ is the local folding of the polypeptide chain into alpha helices and beta sheets elements; the _tertiary structure_ describes the 3D shape of the protein molecule, is composed by a single polypeptide chain backbone with one or more secondary structures and reveals very important functional and chemical properties of the protein; the _quaternary structure_ is the association of several protein chains. 

In this context, we make use of a folding algorithm that exploits deep neural networks techniques to predict the protein tertiary structure from its amino acid sequence.

## HHBlits <a name="hhblits"></a>

We start from the amino acid sequences extracted from the FASTA files obtained as a result of the gene fusion step. The second step is to generate alignments using the tool HHBlits from HHSuite.

One of the main processes in computational biology consists of building protein **multiple-sequence alignments** (MSAs) since MSAs are a key intermediate step in the prediction of protein tertiary structure. MSA refers to a sequence alignment of more than two sequences from which sequence homology can be inferred. 

HHBlits is one of the most important tools to generate multiple sequence alignments. The starting point is to convert the sequences to a condensed representation of a MSA called **profile hidden Markov model** (HMM). HMMs are similar to sequence profiles, in the sense that they transform MSA into _position-specific scoring systems_: profile-profile comparison methods are preferred to sequence-sequence comparison since profiles give much more information than the simple sequence and they are more powerful.  In sequence profiles, each of the 20 amino acids is assigned with a score for each position in the query sequence: this score corresponds to its frequency in that position in the original alignment. These frequencies can be interpreted as probabilities of seen that amino acid in new related proteins. Profile HMM provide, in addition to the amino acid frequencies, also information about the frequencies of insertions and deletions at each columns.

## ProSPr <a name="prospr"></a>

# Models <a name="models"></a>

# Results <a name="results"></a>
