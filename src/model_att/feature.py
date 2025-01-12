import utils
import torch
import consts
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from model_base.base import BaseFeatureExtractor


class FeatureExtractor(BaseFeatureExtractor):
    def __init__(
            self,
            output_dir,
            num_BERT_layers,
            use_cache=True):
        super().__init__(output_dir=output_dir, use_cache=use_cache)
        self.num_BERT_layers = num_BERT_layers

    def _get_model_outputs(self, marked_sents):
        input_idmasks_spans_batches = self._batchify(marked_sents)
        with torch.no_grad():
            for input_ids_batch, input_masks_batch, _ in tqdm(input_idmasks_spans_batches, ncols=100, desc='inference'):
                model_output = consts.LM_MODEL(input_ids_batch, attention_mask=input_masks_batch,
                                               output_hidden_states=False,
                                               output_attentions=True,
                                               return_dict=True)
                batch_attentions = model_output.attentions  # layers, [batch_size, num_heads, seqlen, seqlen]
                batch_attentions = torch.stack(batch_attentions, dim=0)  # [layers, batch_size, num_heads, seqlen, seqlen]
                batch_size, seq_len = input_ids_batch.shape
                batch_attentions = batch_attentions.transpose(0, 1)[:, :self.num_BERT_layers, :, 1:, 1:]
                for i in range(batch_size):
                    yield batch_attentions[i].detach().cpu().numpy()

    def _get_span_attenionmap(self, model_output_dict, l, r):
        """ get feature vector for a span

        Args:
            model_output_dict: a return dict from _get_model_outputs
            l: the left index of the span
            r: the right index of the span
        """
        attention_output = model_output_dict
        return attention_output[:, :, l: r + 1, l: r + 1].copy()

    # def generate_train_instances(self, path_sampled_docs, max_num_docs=None):
    #     utils.Log.info(f'Generating training instances: {path_sampled_docs}')
    #     path_sampled_docs = Path(path_sampled_docs)
    #     path_prefix = 'train.' + f'{max_num_docs}docs.' * (max_num_docs is not None)
    #     path_output = (self.output_dir / path_sampled_docs.name.replace('sampled.', path_prefix)).with_suffix('.pk')
    #     if self.use_cache and utils.IO.is_valid_file(path_output):
    #         print(f'[Feature] Use cache: {path_output}')
    #         return path_output

    #     print(f'Loading: {path_sampled_docs}...', end='')
    #     sampled_docs = utils.OrJsonLine.load(path_sampled_docs)
    #     sampled_docs = sampled_docs[:max_num_docs] if max_num_docs is not None else sampled_docs
    #     print('OK!')
    #     marked_sents = [sent for doc in sampled_docs for sent in doc['sents']]
    #     marked_sents = sorted(marked_sents, key=lambda s: len(s['ids']), reverse=True)
    #     model_outputs = self._get_model_outputs(marked_sents)

    #     train_instances = []
    #     for i, model_output_dict in tqdm(enumerate(model_outputs), ncols=100, total=len(marked_sents), desc='[Feature] Generate train instances'):
    #         marked_sent = marked_sents[i]
    #         word_idxs = marked_sent['widxs']
    #         swidx2widx = {swidx: widx for widx, swidx in enumerate(word_idxs)}
    #         swidx2widx.update({len(marked_sent['ids']): len(swidx2widx)})

    #         for l_idx, r_idx in marked_sent['pos_spans']:
    #             wl_idx, wr_idx = swidx2widx[l_idx], swidx2widx[r_idx + 1] - 1
    #             spanlen = wr_idx - wl_idx + 1
    #             if spanlen > consts.MAX_WORD_GRAM:
    #                 continue
    #             assert spanlen > 0

    #             positive_span_attentionmap = self._get_span_attenionmap(model_output_dict, l_idx, r_idx)
    #             positive_instance = (1, spanlen, positive_span_attentionmap, marked_sent['ids'][l_idx: r_idx + 1])
    #             train_instances.append(positive_instance)

    #         for negative_l_idx, negative_r_idx in marked_sent['neg_spans']:
    #             negative_wl_idx, negative_wr_idx = swidx2widx[negative_l_idx], swidx2widx[negative_r_idx + 1] - 1
    #             negative_spanlen = negative_wr_idx - negative_wl_idx + 1
    #             negative_span_attentionmap = self._get_span_attenionmap(model_output_dict, negative_l_idx, negative_r_idx)
    #             negative_instance = (0, negative_spanlen, negative_span_attentionmap, marked_sent['ids'][negative_l_idx: negative_r_idx + 1])
    #             train_instances.append(negative_instance)

    #         del model_output_dict

    #     utils.Pickle.dump(train_instances, path_output)
    #     return path_output
    def generate_train_instances(self, path_sampled_docs, max_num_docs=None):
        # ---------------------------        
        # Desired final numbers
        POS_TARGET = 2_500_000
        NEG_TARGET = 2_500_000
        
        utils.Log.info(f'Generating training instances: {path_sampled_docs}')
        path_sampled_docs = Path(path_sampled_docs)
        path_prefix = 'train.' + f'{max_num_docs}docs.' * (max_num_docs is not None)
        path_output = (
            self.output_dir
            / path_sampled_docs.name.replace('sampled.', path_prefix)
        ).with_suffix('.pk')

        # Use cache if available
        if self.use_cache and utils.IO.is_valid_file(path_output):
            print(f'[Feature] Use cache: {path_output}')
            return path_output

        # ---------------------------
        # 1) Load and Flatten
        # ---------------------------
        print(f'Loading: {path_sampled_docs}...', end='')
        sampled_docs = utils.OrJsonLine.load(path_sampled_docs)
        if max_num_docs is not None:
            sampled_docs = sampled_docs[:max_num_docs]
        print('OK!')

        # Flatten out the sentences
        marked_sents = [sent for doc in sampled_docs for sent in doc['sents']]
        print(f"Total sentences available: {len(marked_sents):,}")

        # ---------------------------
        # 2) Shuffle and Subset Sentences
        # ---------------------------
        random.shuffle(marked_sents)

        # We might not need all sentences if we only want 5M total instances.
        # But we don't know how many each sentence will produce.
        # Strategy: pick a subset that is "likely" to yield enough.
        # E.g., pick top X% or a specific number of sents if you have a sense of yield.
        # Here's a simple example: take at most 2 million sentences.
        # Adjust as needed.
        max_sents_for_processing = 2_000_000  # or some other cutoff
        if len(marked_sents) > max_sents_for_processing:
            marked_sents = marked_sents[:max_sents_for_processing]
        print(f"Subsampled to {len(marked_sents):,} sentences (pre-process).")

        # Sort sentences by length (descending) if desired (optional step).
        # If you still want the sort-by-length property, do it now or skip it
        marked_sents = sorted(marked_sents, key=lambda s: len(s['ids']), reverse=True)

        # ---------------------------
        # 3) Obtain Model Outputs
        # ---------------------------
        # We process only the (sub)sampled sentences
        model_outputs = self._get_model_outputs(marked_sents)

        # ---------------------------
        # 4) Generate Pos/Neg Instances
        # ---------------------------
        positive_instances = []
        negative_instances = []

        for i, model_output_dict in tqdm(
            enumerate(model_outputs),
            ncols=100,
            total=len(marked_sents),
            desc='[Feature] Generate train instances'
        ):
            marked_sent = marked_sents[i]
            word_idxs = marked_sent['widxs']
            swidx2widx = {swidx: widx for widx, swidx in enumerate(word_idxs)}
            swidx2widx[len(marked_sent['ids'])] = len(swidx2widx)  # boundary

            # Positive spans
            for l_idx, r_idx in marked_sent['pos_spans']:
                wl_idx, wr_idx = swidx2widx[l_idx], swidx2widx[r_idx + 1] - 1
                spanlen = wr_idx - wl_idx + 1
                if spanlen > consts.MAX_WORD_GRAM:
                    continue

                positive_span_attentionmap = self._get_span_attenionmap(
                    model_output_dict, l_idx, r_idx
                )
                positive_instance = (
                    1,  # label
                    spanlen,
                    positive_span_attentionmap,
                    marked_sent['ids'][l_idx : r_idx + 1]
                )
                positive_instances.append(positive_instance)

            # Negative spans
            for neg_l_idx, neg_r_idx in marked_sent['neg_spans']:
                neg_wl_idx, neg_wr_idx = swidx2widx[neg_l_idx], swidx2widx[neg_r_idx + 1] - 1
                neg_spanlen = neg_wr_idx - neg_wl_idx + 1

                negative_span_attentionmap = self._get_span_attenionmap(
                    model_output_dict, neg_l_idx, neg_r_idx
                )
                negative_instance = (
                    0,  # label
                    neg_spanlen,
                    negative_span_attentionmap,
                    marked_sent['ids'][neg_l_idx : neg_r_idx + 1]
                )
                negative_instances.append(negative_instance)

            del model_output_dict

        print(f"Collected {len(positive_instances):,} positive and {len(negative_instances):,} negative instances.")

        # ---------------------------
        # 5) Final Sampling to 5M
        # ---------------------------
        # We want 2.5M positive, 2.5M negative if possible
        random.shuffle(positive_instances)
        random.shuffle(negative_instances)

        pos_keep = min(len(positive_instances), POS_TARGET)
        neg_keep = min(len(negative_instances), NEG_TARGET)

        final_pos = positive_instances[:pos_keep]
        final_neg = negative_instances[:neg_keep]

        final_instances = final_pos + final_neg
        random.shuffle(final_instances)  # interleave positives & negatives

        print(f"Final: {len(final_pos):,} positives + {len(final_neg):,} negatives = {len(final_instances):,} total.")

        # ---------------------------
        # 6) Save the Final Instances
        # ---------------------------
        utils.Pickle.dump(final_instances, path_output)
        print(f"[Feature] Saved to {path_output}")

        return path_output

    
    def generate_predict_docs(self, path_marked_corpus, max_num_docs=None):
        utils.Log.info(f'Generating prediction instances: {path_marked_corpus}')
        path_marked_corpus = Path(path_marked_corpus)

        test_feature_dir = self.output_dir.parent.parent / 'LM_output_for_prediction' / f'Attmap.{consts.LM_NAME_SUFFIX}.{self.num_BERT_layers}layers'
        test_feature_dir.mkdir(exist_ok=True, parents=True)

        predict_name = 'predict.batch.float16.' + f'{max_num_docs}docs.' * (max_num_docs is not None)
        path_output = (test_feature_dir / path_marked_corpus.name.replace('marked.', predict_name)).with_suffix('.pk')
        print(path_output)
        if self.use_cache and utils.IO.is_valid_file(path_output):
            print(f'[FeatureExtractor] Use cache: {path_output}')
            return path_output

        marked_docs = utils.JsonLine.load(path_marked_corpus)
        marked_docs = marked_docs[:max_num_docs] if max_num_docs is not None else marked_docs
        marked_sents = [sent for doc in marked_docs for sent in doc['sents']]
        sorted_i_sents = sorted(list(enumerate(marked_sents)), key=lambda tup: len(tup[1]['ids']), reverse=True)
        marked_sents = [sent for i, sent in sorted_i_sents]
        sorted_raw_indices = [i for i, sent in sorted_i_sents]
        rawidx2newidx = {rawidx: newidx for newidx, rawidx in enumerate(sorted_raw_indices)}

        model_outputs = self._get_model_outputs(marked_sents)

        predict_instances = []
        for i, model_output_dict in tqdm(enumerate(model_outputs), ncols=100, total=len(marked_sents), desc='Generate predict instances'):
            marked_sent = marked_sents[i]
            word_idxs = marked_sent['widxs']

            spans = []
            possible_spans = utils.get_possible_spans(word_idxs, len(marked_sent['ids']), consts.MAX_WORD_GRAM,
                                                      consts.MAX_SUBWORD_GRAM)
            for l_idx, r_idx in possible_spans:
                spanlen = r_idx - l_idx + 1
                spans.append((l_idx, r_idx, spanlen))
            predict_instance = {
                'spans': spans,
                'ids': marked_sent['ids'],
                'attmap': model_output_dict.astype(np.float16)
            }

            predict_instances.append(predict_instance)
            del model_output_dict
        predict_instances = [predict_instances[rawidx2newidx[rawidx]] for rawidx in range(len(predict_instances))]

        # pack predict instances into predict docs
        num_sents_per_doc = [len(doc['sents']) for doc in marked_docs]
        assert len(predict_instances) == sum(num_sents_per_doc)
        pointer = 0
        predict_docs = []
        for doci, num_sents in enumerate(num_sents_per_doc):
            predict_docs.append({
                '_id_': marked_docs[doci]['_id_'],
                'sents': predict_instances[pointer: pointer + num_sents]})
            pointer += num_sents
        assert pointer == len(predict_instances)
        assert len(predict_docs) == len(marked_docs)

        utils.Pickle.dump(predict_docs, path_output)

        return path_output
