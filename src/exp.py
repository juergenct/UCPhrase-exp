import torch
import evaluate
import utils
import os
import gc
import consts
import model_att
import model_emb
import model_base
from tqdm import tqdm
from consts import ARGS
from pathlib import Path
from preprocess import Preprocessor
from preprocess import BaseAnnotator
from preprocess import WikiAnnotator
from preprocess import CoreAnnotator
import nltk

nltk.download('wordnet')

os.environ['HF_HOME'] = '/fibus/fs1/0f/cyh1826/wt/ucphrase/huggingface' # TUHH HPC

class Experiment:
    # rootdir = Path('../experiments')
    # rootdir.mkdir(exist_ok=True)

    # def __init__(self, rootdir='/mnt/nvme01/UCPhrase_JT/experiments'):
    def __init__(self, rootdir='/fibus/fs1/0f/cyh1826/wt/ucphrase/experiments'): # TUHH HPC
        if rootdir is None:
            rootdir = Path('../experiments')
        self.rootdir = Path(rootdir)
        self.rootdir.mkdir(exist_ok=True)
        self.data_config = consts.DATA_CONFIG
        self.path_model_config = consts.PATH_MODEL_CONFIG
        self.config = utils.Json.load(self.path_model_config)
        self.config.update(self.data_config.todict())

        # establish experiment folder
        self.exp_name = f'{consts.DIR_DATA.stem}-{consts.LM_NAME_SUFFIX}-{self.path_model_config.stem}'
        if ARGS.exp_prefix:
            self.exp_name += f'.{ARGS.exp_prefix}'
        self.dir_exp = self.rootdir / self.exp_name
        self.dir_exp.mkdir(exist_ok=True)
        utils.Json.dump(self.config, self.dir_exp / 'config.json')
        print(f'Experiment outputs will be saved to {self.dir_exp}')

        # preprocessor
        self.train_preprocessor = Preprocessor(
            path_corpus=self.data_config.path_train,
            num_cores=consts.NUM_CORES,
            use_cache=True
        )

        # annotator (supervision) # Core annotator is used with CNN model, Wiki annotator is used with emb model --> Core annotator
        self.train_annotator: BaseAnnotator = {
            'wiki': WikiAnnotator(
                use_cache=True,
                preprocessor=self.train_preprocessor,
                path_standard_phrase=self.data_config.path_phrase
            ),
            'core': CoreAnnotator(
                use_cache=True,
                preprocessor=self.train_preprocessor
            )
        }[self.config['annotator']]

        # model
        model_prefix = '.' + ARGS.model_prefix if ARGS.model_prefix else ''
        model_dir = self.dir_exp / f'model{model_prefix}'
        if self.config['model'] == 'CNN': # Standard is to use the CNN model!
            model = model_att.AttmapModel(
                model_dir=model_dir,
                max_num_subwords=consts.MAX_SUBWORD_GRAM,
                num_BERT_layers=self.config['num_lm_layers'])
            self.trainer = model_att.AttmapTrainer(
                model=model)
        elif self.config['model'] == 'emb':
            model = model_emb.EmbedModel(
                model_dir=model_dir,
                finetune=self.config['finetune']
            )
            self.trainer = model_emb.EmbedTrainer(
                model=model
            )

    def train(self, num_epochs=5):
        self.train_preprocessor.tokenize_corpus()
        self.train_annotator.mark_corpus()
        path_sampled_train_data = self.train_annotator.sample_train_data()
        self.trainer.train(path_sampled_train_data=path_sampled_train_data, num_epochs=num_epochs)

    def select_best_epoch(self):
        paths_ckpt = [p for p in self.trainer.output_dir.iterdir() if p.suffix == '.ckpt']
        best_epoch = None
        best_valid_f1 = 0.0
        for p in paths_ckpt:
            ckpt = torch.load(p, map_location='cpu')
            if ckpt['valid_f1'] > best_valid_f1:
                best_valid_f1 = ckpt['valid_f1']
                best_epoch = ckpt['epoch']
        utils.Log.info(f'Best epoch: {best_epoch}. F1: {best_valid_f1}')
        return best_epoch

    def predict(self, epoch, for_tagging=False):
        test_preprocessor = None
        test_preprocessor = Preprocessor(
            path_corpus=self.data_config.path_test,
            num_cores=consts.NUM_CORES,
            use_cache=True)

        test_preprocessor.tokenize_corpus()

        ''' Model Predict '''
        dir_prefix = 'tagging.' if for_tagging else 'kpcand.'
        dir_predict = self.trainer.output_dir / f'{dir_prefix}predict.epoch-{epoch}'
        path_ckpt = self.trainer.output_dir / f'epoch-{epoch}.ckpt'
        model: model_base.BaseModel = model_base.BaseModel.load_ckpt(path_ckpt).eval().to(consts.DEVICE)

        ## Do prediction in chunks
        n_chunks = 80
        # Use the getter to retrieve the path.
        id_path = test_preprocessor.get_path_tokenized_id_corpus()
        # For test data, we assume there's only one file.
        if isinstance(id_path, list):
            id_path = id_path[0]
        all_docs = utils.OrJsonLine.load(id_path)
        # all_docs = utils.OrJsonLine.load(test_preprocessor.path_tokenized_id_corpus)
        total_docs = len(all_docs)
        chunk_size = max(1, (total_docs + n_chunks - 1) // n_chunks)
        print(f"[Predict] Splitting {total_docs} docs into {n_chunks} chunks.")

        dir_predict.mkdir(exist_ok=True)
        chunk_pred_paths = []

        dir_decoded = self.trainer.output_dir / f'{dir_prefix}decoded.epoch-{epoch}'
        dir_decoded.mkdir(exist_ok=True)

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            if start_idx >= total_docs:
                break
            end_idx = min(start_idx + chunk_size, total_docs)
            chunk_docs = all_docs[start_idx:end_idx]

            chunk_file = dir_predict / f'chunk_{chunk_idx}.jsonl'
            utils.OrJsonLine.dump(chunk_docs, chunk_file)

            chunk_pred_path = model.predict(
                path_tokenized_id_corpus=chunk_file,
                dir_output=dir_predict,
                batch_size=1024,
                use_cache=True
            )
            print(f"[Predict] Finished chunk {chunk_idx} ({end_idx - start_idx} docs).")

            # Decode immediately, storing output in dir_decoded (no subdirectories)
            path_decoded_doc2cands = model_base.BaseModel.get_doc2cands(
                path_predicted_docs=chunk_pred_path,
                output_dir=dir_decoded,
                expected_num_cands_per_doc=self.data_config.kp_num_candidates_per_doc,
                use_cache=True,
                use_tqdm=True
            )
            print(f"[Decode] Finished chunk {chunk_idx} decode -> {dir_decoded}")

            # Delete the large .pk file after decoding
            if chunk_pred_path.exists():
                chunk_pred_path.unlink()

            try:
                # Path(f"/mnt/nvme01/UCPhrase_JT/experiments/LM_output_for_prediction/Attmap.roberta-base.3layers/chunk_{chunk_idx}.pk").unlink()
                Path(f"/fibus/fs1/0f/cyh1826/wt/ucphrase/experiments/LM_output_for_prediction/Attmap.roberta-base.3layers/chunk_{chunk_idx}.pk").unlink() # TUHH HPC
            except FileNotFoundError:
                pass

            del chunk_docs
            del chunk_pred_path
            gc.collect()

        print("[Predict] Done with all chunk predictions and decodings.")


if __name__ == '__main__':
    exp = Experiment()
    # exp.train()
    best_epoch = exp.select_best_epoch()
    exp.predict(epoch=best_epoch, for_tagging=False)
