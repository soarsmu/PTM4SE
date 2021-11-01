import re
import os
import csv
import math
import time 
import warnings
warnings.filterwarnings("ignore")

from tqdm          import tqdm
from random        import shuffle, sample, choices
from nltk.corpus   import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem     import PorterStemmer, WordNetLemmatizer
from scipy.stats   import wilcoxon

porter             = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
stopwords_set      = set( stopwords.words('english') )

def _get_words( lines ) :
    words  = list()
    for line in tqdm( lines, desc="Generating random sample from data" )  :
        sents = sent_tokenize( line )
        for sent in sents : 
            for word in word_tokenize( sent ) :
                word = word.lower()
                if word in stopwords_set :
                    continue
                if not re.search( '[^a-zA-Z]', word ) is None :
                    continue
                lem_word  = wordnet_lemmatizer.lemmatize( word )
                words.append( lem_word ) 

    return words
    


def get_data( path, split=False, split_perc=0.5 ) :

    lines = None
    if not re.search( r'.tsv$', path ) is None :
        ## TSV File
        lines = list()
        with open( path ) as f :
            reader = csv.reader( f, delimiter='\t' )
            for line in reader :
                lines.append( line[3] )
    else :
        lines = open( path ).readlines()

    shuffle( lines )
    if not split :
        return _get_words( lines )

    spliter = int( len( lines ) * split_perc  )
    words1 = _get_words( lines[:spliter] )
    words2 = _get_words( lines[spliter:] )
    
    return words1, words2


def wilcox( test1, test2 ) :
    stat, p = wilcoxon( test1, test2 )
    alpha   = 0.05
    return (p,p>alpha )


def _get_freqs( list1, list2 ) :

    all_words = set( list1 + list2 )

    freqs1 = list()
    freqs2 = list()
    for word in tqdm( all_words ):
        freqs1.append( list1.count( word ) )
        freqs2.append( list2.count( word ) )
    return freqs1, freqs2


def compare_samples( test1, test2, sample_description=None, times_to_sample=10, samples=10000 ) :

    truncate = min( len( test1 ), len( test2 ) )
    
    if not samples is None and samples > truncate :
        samples = truncate

    test1_original = test1
    test2_original = test2

    mwu_p_total = 0
    mwu_same    = 0
    mwu_max     = 0
    mwu_min     = 1 

    print()
    print( "Starting tests", end="" )
    if not sample_description is None:
        print( " for ", sample_description )
    else :
        print()
    for sample_attempt in range( times_to_sample ) :
        
        test1 = choices( test1_original, k=samples ) 
        test2 = choices( test2_original, k=samples )
        test1, test2 = _get_freqs( test1, test2 ) 

        ( p, same_dist ) = wilcox( test1, test2 )
        mwu_p_total += p if not math.isnan( p ) else 0
        mwu_same    += 1 if same_dist else 0
        if p < mwu_min :
            mwu_min = p
        if p > mwu_max :
            mwu_max = p
        print( p, "Same Distribution" if same_dist else "NOT same Distribution" )
        

    print()
    print( "*" * 80 )
    if not sample_description is None :
        print( sample_description )
    print( "*" * 80 )

    print( "% SAME       : ", ( mwu_same    / float( times_to_sample ) ) * 100  )
    print( "Max p value  : ",   mwu_max )
    print( "Min p value  : ",   mwu_min )
    print( """
              NOTE: % Same represents the % of tests that reported 
              that the two samples are from the SAME distribution
""" )
    
    print( "*" * 80 )
    print()
    

def main( path, train_name, dev_name ) : 
    
    train_words = get_data( os.path.join( path, train_name   ) ) 
    dev_words   = get_data( os.path.join( path, dev_name     ) )
    
    trainsplit_a, trainsplit_b  = get_data( os.path.join( path, train_name  ), True, 0.80 )

    for ( test1, test2, sample_description )  in [
            ( trainsplit_a , trainsplit_b, "Comparison of splits of Train Data (As typically used used in validation)" ), 
            ( train_words  , dev_words   , "Comparison of Training data vs Development data"         ), 
    ]:
        

        compare_samples( test1, test2, sample_description ) 



if __name__ == '__main__' :
    
    path         = '../datasets/'
    train_name   = 'all_train_data.txt'
    dev_name     = 'all_dev_data.txt'

    main( path, train_name, dev_name )
