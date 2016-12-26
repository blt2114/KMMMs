#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#define BUFF_SIZE 1000000

int kmer_to_idx(char* kmer){
    /* Convert a string representation of a kmer to its hash.
     * 
     * kmer: points to the string representation of our kmer.
     *
     * Returns the integer hash. Or -1 if not a valid sequence.
     */ 
    int idx = 0;
    while (*kmer) {
        // move our bases so far to the left.  In the first iteration, 
        // this does nothing.
        idx<<=2; 
        char b = *kmer++;
        if (b == 'C' || b == 'c') idx += 1;
        else if (b == 'G' || b== 'g') idx += 2; 
        else if (b == 'T' || b== 't') idx += 3;
        else if (!(b == 'A' || b == 'a')) return -1; // e.g. N
    }
    return idx;
}

char* idx_to_kmer(int idx, int K, char* kmer_buff){
    /* Convert a hash to the kmer it represents.
     *
     * K: the length of the kmer
     * kmer_buff: where to write the kmer
     *
     * Returns a pointer to the string.
     */
    
    // We fill the buffer starting with the last base because this 
    // is how it is stored in kmer_to_idx
    int k = K; 
    while (k--) { // in our first iteration, we use index k == (K-1)
        // The last two bits contains the identity of the next base.
        int b= idx %4; 

        if (b == 0) kmer_buff[k]='A';
        else if (b == 1) kmer_buff[k]='C';
        else if (b == 2) kmer_buff[k]='G';
        else kmer_buff[k]='T'; // b == 3
        idx >>=2;
    }
    return kmer_buff;
}

void add_to_kmer_table(char *seq, int *kmer_table, int K){ 
    /* adds the all kmers within a sequence into our kmer count table.
     *
     * seq: points to our kmer
     * kmer_table: integer array of counts.
     * K: the length of our kmers
     */
    char kmer[K+1]; // must include terminator char
    memset(kmer,0,K+1);
    while (*(seq + K-1)){ 
        // We must strncpy so that we null terminate
        strncpy(kmer,seq++,K); 
        int idx = kmer_to_idx(kmer);
        if (idx == -1) continue;
        kmer_table[idx]++;
    }
}

int main(int argc, char** argv){
    if (argc != 3){
        fprintf(stderr,"./create_kmer_table K seqs.fa\n");
        exit(1);
    }
            
    int K = atoi(*++argv);
    char *fn = *++argv;
    FILE *fp = fopen(fn,"r");
    int kmer_table[1<<(2*K)];
    memset(kmer_table,0,1<<(2*K+2)); // initialize memory to 0's
    
    // hold the next line and any sequence that was left over from 
    // reading the last line
    char seq_buff[BUFF_SIZE+K-1];
    memset(seq_buff,0,BUFF_SIZE+K-1); // initialize memory to 0's

    // all lines will be read into the buffer after this left-over portion
    char *line_buff = seq_buff+K-1;
    char *seq_start = NULL; // kmers are built on sequence starting from here.

    // Parse through fasta file to create kmer table.
    while ( fgets(line_buff,BUFF_SIZE,fp)){
        // if the header of sequence, read next line.
        if (line_buff[0] == '>'){
            fgets(line_buff,BUFF_SIZE,fp);
            memset(seq_buff,0,K-1); // new sequence, so wipe recent bases.
            seq_start = line_buff;
        }else{
            seq_start = seq_buff;
        }
        
        add_to_kmer_table(seq_start, kmer_table, K);
        if (line_buff[BUFF_SIZE-1] == '\n'){
            line_buff[strlen(line_buff)-1]='\0'; // eliminate newline 
            // Copy the last K-1 letters into the beginning of the 
            // sequence buffer.
            strncpy(seq_buff,line_buff+BUFF_SIZE-K-1,K-1); 
        }else{
            // if we didn't have a newline, this is one base later.
            strncpy(seq_buff,line_buff+BUFF_SIZE-K,K-1); 
        }
    }

    int kmer_idx = 0;
    char kmer [K+1];
    memset(kmer,0,K+1);
    while (kmer_idx < 1<<(2*K)){
        printf("%s\t%d\n",idx_to_kmer(kmer_idx, K, kmer),kmer_table[kmer_idx]);
        kmer_idx++;
    }
    return 0;
}
