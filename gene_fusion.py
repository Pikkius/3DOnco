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
                    break
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

    def print_fasta(self, Sequence, config):  # config = {nome, Chr5p, Chr3p, Coord5p, Coord3p}
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        with open(self.out_dir + '/' + config['nome'] + '.fa', 'w') as f:
            f.writelines(['> ' + str(config)[1:-1], '\n', Sequence])

    def gene_fusion(self, data, chromosome, bp, flag5):  # 1--> 5, 0-->3
        mask_in = ((data['Chr'] == chromosome) & (data['Start_T'] < bp) & (data['End_T'] > bp))

        list_out = []
        list_gene = []
        for gene in set(data[mask_in]['Gene']):
            mask = mask_in & (data['Gene'] == gene)
            gene_dataframe = data[mask]
            list_gene.append(gene)

            if gene_dataframe.iloc[-1]['Sign'] == '+':
                sign = 0  # sing = 0 se +; 1 se -
            else:
                sign = 1

            if bool(flag5) != bool(sign):  # se è 5+ o 3-

                mask = mask & (data['Start'] < bp)  # Take only bases before break point
                gene_dataframe = data[mask]

                if len(gene_dataframe) == 0:  # Exit early if no CDS is found
                    list_out.append(" ")
                    continue
                if bp > int(gene_dataframe.iloc[-1]['End']):
                    bp = gene_dataframe.iloc[-1]['End']
                else:
                    gene_dataframe = (gene_dataframe.replace(gene_dataframe.iloc[-1]['End'], bp)).reset_index()
                start = gene_dataframe.iloc[0]['Start']
                end = gene_dataframe.iloc[-1]['End']

            else:  # se è 5- o 3+
                mask = mask & (data['End'] > bp)  # Take only bases before break point
                gene_dataframe = data[mask]

                if len(gene_dataframe) == 0:  # Exit early if no CDS is found
                    list_out.append(" ")
                    continue
                if bp < gene_dataframe.iloc[0]['Start']:
                    bp = gene_dataframe.iloc[0]['Start']
                else:
                    gene_dataframe = gene_dataframe.replace(gene_dataframe.iloc[0]['Start'], bp)
                start = gene_dataframe.iloc[0]['Start']
                end = gene_dataframe.iloc[-1]['End']

            gene_dataframe = gene_dataframe.reset_index()

            # Load Chromosome DNA
            DNA = self.retrive_sequence_dna(start - 1, end, chromosome)  # zip version is for files still compressed

            part = []
            n = len(gene_dataframe)

            # Retrive all CDSs sequences
            for i in range(n):
                start_tmp, end_tmp = gene_dataframe.loc[i, ['Start', 'End']].values
                CDS = DNA[start_tmp - start:end_tmp + 1 - start]  # Take the sequence

                if sign == 1: CDS = complementa(CDS)
                part.append(CDS)

            # Unite all CDSs sequences
            if sign == 1:
                part = ''.join(part[::-1])
            else:
                part = ''.join(part)

            list_out.append(part)
        return list_out, list_gene

    def fit(self):
        # ADI ha sempre ragione

        # config = {nome, Chr5p, Chr3p, Coord5p, Coord3p}
        config = {'nome': 0, 'Label': 0,
                  'Chr5p': 0, 'Coord5p': 0, '5pEnsg': 0, '5pStrand': 0,
                  'Chr3p': 0, 'Coord3p': 0, '3pEnsg': 0, '3pStrand': 0,
                  'shift_5': 0, 'shift_3': 0, 'shift_tot': 0, 'flag_end_codon': 0}

        # to_do = list(np.load('/content/drive/MyDrive/Fastas/problems.npy'))
        # to_delete_1 = [255, 256, 308, 373, 376, 410, 411, 412, 414,
        # 415, 416, 417, 418, 441, 442, 454, 502]

        problems = []
        for index, row in self.table.iterrows():
            if str(index) + '_' + row['FusionPair'] not in os.listdir(self.out_dir):
                # if index in to_do:

                config['nome'] = str(index) + '_' + row['FusionPair']
                config['Chr5p'] = row['Chr5p']
                config['Chr3p'] = row['Chr3p']
                config['Coord5p'] = int(row['Coord5p'])
                config['Coord3p'] = int(row['Coord3p'])
                config['5pStrand'] = str(row['5pStrand'])
                config['3pStrand'] = str(row['3pStrand'])
                config['Label'] = int(row['Label'])

                count = 0
                # ANDRE HA CAMBIATO COMBINATIONS CON PRODUCT
                seq_5, gene_5 = self.gene_fusion(self.data, config['Chr5p'], config['Coord5p'], 1)
                seq_3, gene_3 = self.gene_fusion(self.data, config['Chr3p'], config['Coord3p'], 0)

                combinations = list(itertools.product(seq_5, seq_3))
                combinations_genes = list(itertools.product(gene_5, gene_3))

                for result, res_gene in zip(combinations, combinations_genes):
                    a = str(result[0])
                    b = str(result[1])
                    result = a + b

                    config['5pEnsg'] = res_gene[0]
                    config['3pEnsg'] = res_gene[1]
                    config['shift_5'] = len(a) % 3
                    config['shift_3'] = len(b) % 3
                    config['shift_tot'] = len(result) % 3

                    amino_seq, config['flag_end_codon'] = translate(result)

                    if count == 0:
                        print('fatto')
                        self.print_fasta(amino_seq, config)
                    else:
                        # if index in to_delete_1: continue
                        print(f'multi out with {index}')
                        problems.append(index)
                        self.print_fasta(self.out_dir + config['nome'] + '_' + str(count), amino_seq, config)
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
