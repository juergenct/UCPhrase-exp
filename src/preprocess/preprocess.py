import utils
import consts
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


class Preprocessor:

    def __init__(
            self,
            path_corpus,
            num_cores=16,
            use_cache=True):
        self.use_cache = use_cache
        self.num_cores = num_cores

        # establish preprocess folder
        if isinstance(path_corpus, list):
            self.path_corpus_list = [Path(p) for p in path_corpus]
        else:
            self.path_corpus_list = [Path(path_corpus)]
        self.dir_corpus = self.path_corpus_list[0].parent
        self.dir_preprocess = self.dir_corpus / f'preprocess-{consts.LM_NAME_SUFFIX}'
        self.dir_preprocess.mkdir(exist_ok=True)

        # path_tokenized_corpus: wordpieces tokenized with huggingface LM tokenizer
        # path_tokenized_id_corpus: tokenized wordpiece ids with word boundaries
        self.tokenized_corpus_files = []     # For the tokenized text version.
        self.tokenized_id_corpus_files = []    # For the tokenized ids (with word boundaries).
        for p in self.path_corpus_list:
            base_name = p.name  # e.g. "devdata_fold1.train.0.jsonl"
            tokenized_path = self.dir_preprocess / f'tokenized.{base_name}'
            tokenized_id_path = self.dir_preprocess / f'tokenized.id.{base_name}'
            self.tokenized_corpus_files.append(tokenized_path)
            self.tokenized_id_corpus_files.append(tokenized_id_path)

    
    # def _par_tokenize_doc(doc):
    #     docid = doc['_id_']
    #     sents = doc['sents']

    #     # tokenize
    #     # NOTE: add space before each raw sentence to tokenize the first token with GPT_TOKEN for phrase matching
    #     tokenized_sents = [consts.LM_TOKENIZER.tokenize(' ' + s, add_special_tokens=False) for s in sents]
    #     cleaned_tokenized_sents = []
    #     for tokens in tokenized_sents:
    #         tokens_batch = utils.get_batches(tokens, batch_size=consts.MAX_SENT_LEN)
    #         cleaned_tokenized_sents += tokens_batch
    #     tokenized_doc = {'_id_': docid, 'sents': [' '.join(tokens) for tokens in cleaned_tokenized_sents]}

    #     tokenized_id_doc = {'_id_': doc['_id_'], 'sents': []}
    #     for tokens in cleaned_tokenized_sents:
    #         widxs = [i for i, token in enumerate(tokens) if token.startswith(consts.GPT_TOKEN)]  # the indices of start of words
    #         ids = consts.LM_TOKENIZER.convert_tokens_to_ids(tokens)
    #         tokenized_id_doc['sents'].append({'ids': ids, 'widxs': widxs})

    #     return tokenized_doc, tokenized_id_doc
    
    @staticmethod
    def _par_tokenize_doc(doc):
        docid = doc['_id_']
        sents = doc['sents']

        # Tokenize each sentence using the LM tokenizer.
        # (A leading space is added so that the first token is tokenized correctly.)
        tokenized_sents = [consts.LM_TOKENIZER.tokenize(' ' + s, add_special_tokens=False) for s in sents]
        cleaned_tokenized_sents = []
        for tokens in tokenized_sents:
            tokens_batch = utils.get_batches(tokens, batch_size=consts.MAX_SENT_LEN)
            cleaned_tokenized_sents += tokens_batch
        tokenized_doc = {'_id_': docid, 'sents': [' '.join(tokens) for tokens in cleaned_tokenized_sents]}

        tokenized_id_doc = {'_id_': docid, 'sents': []}
        for tokens in cleaned_tokenized_sents:
            # Identify word boundaries (e.g. tokens starting with the special GPT token).
            widxs = [i for i, token in enumerate(tokens) if token.startswith(consts.GPT_TOKEN)]
            ids = consts.LM_TOKENIZER.convert_tokens_to_ids(tokens)
            tokenized_id_doc['sents'].append({'ids': ids, 'widxs': widxs})
        return tokenized_doc, tokenized_id_doc

    # def tokenize_corpus(self):
    #     if self.use_cache and utils.IO.is_valid_file(self.path_tokenized_corpus) and utils.IO.is_valid_file(self.path_tokenized_id_corpus):
    #         print(f'[Preprocessor] Use cache: {self.path_tokenized_corpus}')
    #         return
    #     docs = utils.JsonLine.load(self.path_corpus)
    #     pool = Pool(processes=self.num_cores)
    #     pool_func = pool.imap(func=Preprocessor._par_tokenize_doc, iterable=docs)
    #     doc_tuples = list(tqdm(pool_func, total=len(docs), ncols=100, desc=f'[Tokenize] {self.path_corpus}'))
    #     tokenized_docs = [doc for doc, iddoc in doc_tuples]
    #     tokenized_id_docs = [iddoc for doc, iddoc in doc_tuples]
    #     pool.close()
    #     pool.join()
    #     utils.JsonLine.dump(tokenized_docs, self.path_tokenized_corpus)
    #     utils.JsonLine.dump(tokenized_id_docs, self.path_tokenized_id_corpus)

    def tokenize_corpus(self):
        # Process each training partition file separately.
        for idx, corpus_path in enumerate(self.path_corpus_list):
            tokenized_path = self.tokenized_corpus_files[idx]
            tokenized_id_path = self.tokenized_id_corpus_files[idx]
            if self.use_cache and utils.IO.is_valid_file(tokenized_path) and utils.IO.is_valid_file(tokenized_id_path):
                print(f'[Preprocessor] Use cache: {tokenized_path}')
                continue

            print(f'[Preprocessor] Processing partition: {corpus_path.name}')
            docs = utils.JsonLine.load(corpus_path)
            pool = Pool(processes=self.num_cores)
            pool_func = pool.imap(func=Preprocessor._par_tokenize_doc, iterable=docs)
            doc_tuples = list(tqdm(pool_func, total=len(docs), ncols=100, desc=f'[Tokenize] {corpus_path.name}'))
            tokenized_docs = [doc for doc, iddoc in doc_tuples]
            tokenized_id_docs = [iddoc for doc, iddoc in doc_tuples]
            pool.close()
            pool.join()
            utils.JsonLine.dump(tokenized_docs, tokenized_path)
            utils.JsonLine.dump(tokenized_id_docs, tokenized_id_path)

    def get_path_tokenized_id_corpus(self):
        if len(self.tokenized_id_corpus_files) == 1:
            return self.tokenized_id_corpus_files[0]
        else:
            return self.tokenized_id_corpus_files
