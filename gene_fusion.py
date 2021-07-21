import pandas as pd
import numpy as np
import gzip
import os
import itertools


class load_dataset:

    def __init__(self):
        self.dataframe = pd.DataFrame(columns=['Chr', 'Gene', 'Transcript', 'Start', 'End', 'Sign', 'Start_T', 'End_T'])

    def load(self, path):
        with open(path, 'r') as f:
            for row in f:
                if ('protein_coding' in row) and (row.split('\t')[2] == 'transcript') and (
                        row.split('\t')[1] == 'ensembl' or row.split('\t')[1] == 'ensembl_havana'):  # chiedi a marta
                    trans_features = row.split('\t')[3:5]
                    Trans_id = ((row.split('\t')[8].split(';'))[2].split(' ')[2]).replace('"', '')

                if ('protein_coding' in row) and (row.split('\t')[2] == 'CDS') and (
                        row.split('\t')[1] == 'ensembl' or row.split('\t')[1] == 'ensembl_havana'):  # chiedi a marta
                    features = row.split('\t')
                    chromosome = features[0]
                    gene_id = (features[8].split(' ')[1]).replace('"', '')[:-1]
                    transcript_id = ((features[8].split(';'))[2].split(' ')[2]).replace('"', '')

                    self.dataframe.append([chromosome, gene_id, transcript_id, int(features[3]), int(features[4]),
                                           features[6]] + trans_features)

    def select(self):
        tmp = pd.DataFrame()

        for k, v in self.dataframe.groupby(['Chr', 'Gene']):
            trans = list(set(v['Transcript'].values))
            lunghezze = []
            for i, el in enumerate(trans):
                mask = v['Transcript'] == el
                tot_len = int(v[mask]['End_T'].iloc[-1]) - int(v[mask]['Start_T'].iloc[-1])
                lunghezze.append(tot_len)
            if lunghezze.count(max(lunghezze)) == 1:
                selected_trans = trans[lunghezze.index(max(lunghezze))]
            else:
                trans = np.array(trans)
                max_len = np.argwhere(lunghezze == np.amax(lunghezze)).flatten().tolist()
                selected_trans = trans[max_len]
                len_trans = []
                for S_Trans in selected_trans:
                    len_trans.append(sum(v['Transcript'] == S_Trans))
                selected_trans = selected_trans[len_trans.index(max(len_trans))]

            mask = v['Transcript'] == selected_trans
            tmp = pd.concat([tmp, v[mask]])
        self.dataframe = tmp

    def save(self, path):
        (self.dataframe.reset_index()).drop(columns='index').to_csv(path, index_label=False)


class gene_fusion:

    def __init__(self, dna_path, out_dir):
        self.dna_path = dna_path
        self.dataset_path = None
        self.out_dir = out_dir
        self.data = None
        self.table = None
        self.config = {'nome': 0, 'Label': 0,
                       'Chr5p': 0, 'Coord5p': 0, '5pEnsg': 0, '5pStrand': False, 'intron_5': False,
                       'Chr3p': 0, 'Coord3p': 0, '3pEnsg': 0, '3pStrand': False, 'intron_3': False,
                       'shift_5': 0, 'shift_3': 0, 'shift_tot': 0, 'flag_end_codon': 0}

    def load_tablet(self):
        self.table = pd.read_csv(self.dataset_path, sep='\t', index_col=0)

    def load_data(self, data_path):
        # Load dataset
        self.data = pd.read_csv(data_path)
        self.data['Chr'] = self.data['Chr'].astype(str)
        self.data['Gene'] = self.data['Gene'].astype(str)
        self.data['Start'] = self.data['Start'].astype(int)
        self.data['Transcript'] = self.data['Transcript'].astype(str)
        self.data['End'] = self.data['End'].astype(int)
        self.data['Sign'] = self.data['Sign'].astype(str)
        self.data['Start_T'] = self.data['Start_T'].astype(int)
        self.data['End_T'] = self.data['End_T'].astype(int)

    def retrive_sequence_dna(self, start, end, chromosome):
        name_file = f'{self.dna_path}/Homo_sapiens.GRCh37.dna.chromosome.{str(chromosome)}.fa'

        f = open(name_file, 'r')
        flag_zip = False
        try:
            next(f)
            gzip_fd = f
        except:
            f.close()
            f = open(name_file, 'rb')
            gzip_fd = gzip.GzipFile(fileobj=f)
            next(gzip_fd)
            flag_zip = True
        cnt = 0
        seq = []
        flag_start = False
        for row in gzip_fd:
            if flag_zip:
                row = row.decode("utf-8").strip()
            else:
                row = row.strip()

            if flag_start:
                if end - cnt < len(row):
                    seq.append(row[:end - cnt])
                    break #end found
                else:
                    seq.append(row)
                    cnt = cnt + len(row)
            else:
                if start - cnt < len(row):
                    diff = start - cnt
                    seq.append(row[diff:])
                    flag_start = True  # start found
                    cnt = cnt + len(row)
                else:
                    cnt = cnt + len(row)
        f.close()
        gzip_fd.close()
        return ''.join(seq)

    def print_fasta(self, Sequence, count=0):  # config = {nome, Chr5p, Chr3p, Coord5p, Coord3p}
        out_dir = self.out_dir + self.config['nome'] + '_' + str(count) if count!=0 else self.out_dir + self.config['nome']
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        with open(out_dir + '/' + self.config['nome'] + '.fa', 'w') as f:
            f.writelines(['> ' + str(self.config)[1:-1], '\n', Sequence])

    def filter_dataframe(self, gene, flag5):

        chromosome = self.config['Chr5p'] if flag5 else self.config['Chr3p']
        bp = self.config['Coord5p'] if flag5 else self.config['Coord3p']
        mask= ((self.data['Chr'] == chromosome) & (self.data['Gene'] == gene)
               & (self.data['Start_T'] < bp) & (self.data['End_T'] > bp))

        gene_dataframe = self.data[mask]

        if flag5:
            self.config['5pStrand'] = False if gene_dataframe.iloc[-1]['Sign'] == '+' else True  # sing = 0 se +; 1 se -
            sign = self.config['5pStrand']
        else:
            self.config['3pStrand'] = False if gene_dataframe.iloc[-1]['Sign'] == '+' else True  # sing = 0 se +; 1 se -
            sign = self.config['3pStrand']

        if flag5 != sign:  # se è 5+ o 3-

            mask = mask & (self.data['Start'] < bp)  # Take only bases before break point
            gene_dataframe = self.data[mask]

            if bp > int(gene_dataframe.iloc[-1]['End']):
                if flag5:
                    self.config['intron_5'] = True
                else:
                    self.config['intron_3'] = True

            else:
                gene_dataframe = (gene_dataframe.replace(gene_dataframe.iloc[-1]['End'], bp)).reset_index()

        else:  # se è 5- o 3+
            mask = mask & (self.data['End'] > bp)  # Take only bases before break point
            gene_dataframe = self.data[mask]

            if bp < gene_dataframe.iloc[0]['Start']:
                if flag5:
                    self.config['intron_5'] = True
                else:
                    self.config['intron_3'] = True
            else:
                gene_dataframe = gene_dataframe.replace(gene_dataframe.iloc[0]['Start'], bp)

        return gene_dataframe.reset_index()

    def check_multiple_gene(self, flag5):
        chromosome = self.config['Chr5p'] if flag5 else self.config['Chr3p']
        bp = self.config['Coord5p'] if flag5 else self.config['Coord3p']
        mask_in = ((self.data['Chr'] == chromosome) & (self.data['Start_T'] < bp) & (self.data['End_T'] > bp))
        return set(self.data[mask_in]['Gene'])

    def gene_seq_retrival(self, gene, introne, flag5):  # 1--> 5, 0-->3

        chromosome = self.config['Chr5p'] if flag5 else self.config['Chr3p']

        gene_dataframe = self.filter_dataframe(gene, flag5)
        if len(gene_dataframe) == 0:
            return " "

        sign = self.config['5pStrand'] if flag5 else self.config['3pStrand']

        if introne:
            if flag5:
                DNA = self.retrive_sequence_dna(gene_dataframe.iloc[0]['Start'] - 1, self.config['Coord5p'],
                                                chromosome)  # zip version is for files still compressed
                start = gene_dataframe.iloc[0]['Start']
                end = self.config['Coord5p']
            else:
                DNA = self.retrive_sequence_dna(self.config['Coord3p'] - 1, gene_dataframe.iloc[-1]['End'],
                                                chromosome)  # zip version is for files still compressed
                start = self.config['Coord3p']
                end = gene_dataframe.iloc[-1]['End']
        # Load Chromosome DNA
        else:
            DNA = self.retrive_sequence_dna(gene_dataframe.iloc[0]['Start'] - 1, gene_dataframe.iloc[-1]['End'],
                                        chromosome)  # zip version is for files still compressed
            start = gene_dataframe.iloc[0]['Start']
            end = gene_dataframe.iloc[-1]['End']

        part = []
        n = len(gene_dataframe)

        # Retrive all CDSs sequences
        if introne and not(flag5):
            part.append(DNA[:gene_dataframe.loc[0, ['Start']].values + 1 - start])
        for i in range(n):
            start_tmp, end_tmp = gene_dataframe.loc[i, ['Start', 'End']].values
            CDS = DNA[start_tmp - start:end_tmp + 1 - start]  # Take the sequence

            if sign: CDS = complementa(CDS)
            part.append(CDS)
        if introne and flag5:
            part.append(DNA[gene_dataframe.loc[n, ['end']].values - start:])

        # Unite all CDSs sequences
        if sign:
            part = ''.join(part[::-1])
        else:
            part = ''.join(part)

        return part

    def fit(self):
        # ADI ha sempre ragione

        for index, row in self.table.iterrows():
            if str(index) + '_' + row['FusionPair'] not in os.listdir(self.out_dir):
                # if index in to_do:

                self.config['nome'] = str(index) + '_' + row['FusionPair']
                self.config['Chr5p'] = row['Chr5p']
                self.config['Chr3p'] = row['Chr3p']
                self.config['Coord5p'] = int(row['Coord5p'])
                self.config['Coord3p'] = int(row['Coord3p'])
                self.config['5pStrand'] = str(row['5pStrand'])
                self.config['3pStrand'] = str(row['3pStrand'])
                self.config['Label'] = int(row['Label'])

                count = 0
                # ANDRE HA CAMBIATO COMBINATIONS CON PRODUCT

                gene_5 = self.check_multiple_gene(True)
                gene_3 = self.check_multiple_gene(False)

                combinations_genes = list(itertools.product(gene_5, gene_3))

                for res_gene in combinations_genes:
                    gene_dataframe_5 = self.filter_dataframe(res_gene[0], True)
                    gene_dataframe_3 = self.filter_dataframe(res_gene[1], False)
                    if self.config['intron_5'] != self.config['intron_3']:
                        a = self.gene_seq_retrival(gene_dataframe_5, self.config['intron_5'], True)
                        b = self.gene_seq_retrival(gene_dataframe_3, self.config['intron_3'], False)
                    else:
                        a = self.gene_seq_retrival(gene_dataframe_5, False, True)
                        b = self.gene_seq_retrival(gene_dataframe_3, False, False)

                    result = a + b

                    self.config['5pEnsg'] = res_gene[0]
                    self.config['3pEnsg'] = res_gene[1]
                    self.config['shift_5'] = len(a) % 3
                    self.config['shift_3'] = len(b) % 3
                    self.config['shift_tot'] = len(result) % 3

                    amino_seq, self.config['flag_end_codon'] = translate(result)

                    if count == 0:
                        print('fatto')
                        self.print_fasta(amino_seq)
                    else:
                        # if index in to_delete_1: continue
                        print(f'multi out with {index}')
                        self.print_fasta(amino_seq, count)
                    count += 1


def complementa(sequence):
    compl = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    sequence = sequence[::-1]  # inverti sequenza
    sequence = list(map(lambda c: compl[c], sequence))  # complementa basi
    return ''.join(sequence)


def translate(seq):
    table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }
    protein = " "
    seq = seq.replace(" ", "")
    n = len(seq)
    resto = (n % 3)

    if resto != 0:
        seq = seq[:n - resto]  # si taglia l'ultima base non completa

    flag_stop_codon = 0
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        aminoacid = table[codon]
        if aminoacid != '_':
            protein += aminoacid
        else:
            flag_stop_codon = 1
            break
    return protein, flag_stop_codon
