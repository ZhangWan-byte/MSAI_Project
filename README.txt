# SAbDab Data Processing

- 1_cdr_seq_IMGT
    - ctrl c+v from SAbDab CDR sequence search page，get all heavy/light chains of PDBs and their CDRs
- 2_data_preprocessing
    - Criterion
        - resolution < 4A
        - have VH, VL and VA
    - sequence_pairs.json has null Hseq
        - capitalize all sequence ids; if the chain is within PDB, then get lower case, else upper case;
- 3_seqs_structs
    - skip processing non-common AAs, such as HOH;
    - some atoms are missed (e.g. C, O), substitute them with first atom in this AA;
        - theoretically it's possible to calculate relative positions of N/CA/C/O in AA, however it's not known how effective it is;
    - chain matching problem
        - capitalize all sequence ids; if the chain is within PDB, then get lower case, else upper case;
            - [Possible Issue] there are Fab or FEE indicating ab and EE are heavy/light chains. However, chains could be A/B or e in PDB file.
	After visualising a crystal structure, 7pgb has multiple symmetrical structure, so it has to be ensured to distinguish between Upper and Lower case letter.
	Otherwise, it could be Mn but mistakened as MN;
    - delete PDBs without antigen chain;
        - PDBs that have CDR sequences but no corresponding antigechains should be removed;
- **TODO**
    - **Ab-Ag interface: find the nearest 48 AA as epitope；**
    - **sequence duplicate removal: repetitive rates of H/L/A above 95% should be removed**