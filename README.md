# Blackbox-attack

This is the code we used in the following paper
>[A Closer Look into the Robustness of Neural Dependency Parsers
Using Better Adversarial Examples](https://aclanthology.org/2021.findings-acl.207.pdf)

>Findings of ACL 2021


## Requirements

Python 3.6, PyTorch >=1.3.1, ...

## Data format
For the data format used in our implementation, please read the code (conllx
 format)
 
## Running the experiments
First to the experiments folder:

    cd experiments

### Dependency Parsing
To train a Stack-Pointer parser, simply run

    ./scripts/train_stackptr.sh
Remeber to setup the paths for data and embeddings.

To train a Deep BiAffine parser, simply run

    ./scripts/train_biaf.sh
Again, remember to setup the paths for data and embeddings.

### generate adversarial samples offline
First to the adversary folder:

    cd adversary
    
+ get the train_vocab.json or test_vocab.json from training set or test set


    ./scripts/build_vocab_from_conll.py
 
+ get the candidate words of each token in **training set** with
sememe-based method


     ./scripts/lemma.py  
     ./scripts/gen_candidates.py
     
+ get the candidate words of each token in **test set** with
synonym-based method(note: input vocab is test_vocab.json, not training_vocab
.json)


    ./scripts/gen_synonym.py 
    
+  get the candidate words of each token in **test set** with
bert-based method

    ./scripts/gen_mlm_cands.sh
    
+ adjust the embedding according to <https://github.com/nmrksic/counter
-fitting> Maybe,you need to compute the similarity Matrix using compute_nn.sh and
   merge_nn.sh

At last, you need to run the scirpt ./scripts/preprocess.py to get the total
 adversarial cache under **test set** or **training set** 

### adversarial attacking

    python ./pipeline.py --gpu 0 ./config/pipe.json ./config/stanfordtag.json
     output_dir
     
### ensemble 
if you want to attack the ensembled model, you need to add " --ensemble
 " in ./scripts/pipeline.py
 
### adversarial training
To train a Deep BiAffine parser, simply run

    ./scripts/train_biaf.sh
But, remember to setup "add_path" for adding adversarial sample. (you can
 find  adversarial gold samples after you run "adversarial attacking")


 