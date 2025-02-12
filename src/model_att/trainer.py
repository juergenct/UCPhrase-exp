import utils
import torch
import random
import gc
from consts import DEVICE
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm
from model_att.model import AttmapModel


class AttmapTrainLoader:
    def __init__(self, random_seed=42, max_num_subwords=10):
        self.random_seed = random_seed
        self.max_num_subwords = max_num_subwords

    def get_batch_size(self):
        return 2048

    def get_loader(self, instances, is_train=True):
        batch_size = self.get_batch_size()
        print(f'original training instances: {len(instances)}')
        instances = [instance for instance in instances if instance[2].shape[-1] <= self.max_num_subwords]
        print(f'useful training instances: {len(instances)}')
        gtlabels = [instance[0] for instance in instances]
        spanlens = [instance[1] for instance in instances]
        attention_maps = [instance[2] for instance in instances]
        gtlabels = torch.tensor(gtlabels, dtype=torch.long)
        spanlens = torch.tensor(spanlens, dtype=torch.long)
        attmap_features = AttmapModel.pad_attention_maps(attention_maps, max_num_subwords=self.max_num_subwords)

        dataset = TensorDataset(attmap_features, gtlabels)
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def load_train_data(self, filepath, sample_ratio=-1):
        print('Loading training data...',)
        instances = utils.Pickle.load(filepath)
        print(f'OK! {len(instances)} training instances')

        if sample_ratio > 0.0:
            assert sample_ratio < 1.0
            num_instances = int(sample_ratio * len(instances))
            instances = random.choices(instances, k=num_instances)
            print(f'[Trainer] Sampled {len(instances)} instances.')

        train_instances, valid_instances = train_test_split(instances, test_size=0.1, shuffle=True, random_state=self.random_seed)
        return self.get_loader(train_instances), self.get_loader(valid_instances)


class AttmapTrainer:
    def __init__(self, model: AttmapModel, sample_ratio=-1):
        self.sample_ratio = sample_ratio

        self.model = model.to(DEVICE)
        self.output_dir = model.model_dir
        self.train_loader = AttmapTrainLoader()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        model_config_path = self.output_dir / 'model_config.json'
        utils.Json.dump(self.model.config, model_config_path)

    def train(self, path_sampled_train_data, num_epochs=5):
        """
        Args:
            path_sampled_train_data: 
                A list of file paths, each containing a subset of the training dataset. 
                e.g. ["train_part0.pk", "train_part1.pk", ...]
            num_epochs: Number of epochs to train.
        """
        path_train_data = None
        utils.Log.info('Feature extraction...')
        # This returns something like ["train_part0.pk", "train_part1.pk", ...]
        path_train_data = self.model.feature_extractor.generate_train_instances(path_sampled_train_data)

        utils.Log.info(f'Start training on parts: {path_train_data}')

        best_epoch = -1
        best_valid_f1 = -1.0

        for epoch in range(1, num_epochs + 1):
            utils.Log.info(f'Epoch [{epoch} / {num_epochs}]')

            # Put the model in training mode
            self.model.train()
            epoch_loss = 0.0
            total_train_samples = 0

            # We'll store valid_data from the **last** chunk only
            last_valid_data = None

            # -- 1) Train on Each Part/Chunk --
            for part_idx, part_path in enumerate(path_train_data):
                utils.Log.info(f'[Epoch {epoch}] Loading training data (part {part_idx+1}/{len(path_train_data)}): {part_path}')
                
                # Load the chunk's train_data and valid_data
                # (Depending on your loader logic, valid_data could be empty or None
                #  if you only really want a single chunk for validation.)
                train_data, valid_data = self.train_loader.load_train_data(
                    part_path,
                    sample_ratio=self.sample_ratio
                )

                num_train = len(train_data)
                total_train_samples += num_train

                # --- Train on this chunk ---
                for attmap_features, gtlabels in tqdm(train_data, total=num_train, ncols=100):
                    self.model.zero_grad()
                    gtlabels = gtlabels.to(DEVICE)
                    attmap_features = attmap_features.to(DEVICE)
                    
                    batch_loss = self.model.get_loss(gtlabels, attmap_features)
                    epoch_loss += batch_loss.item()

                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                # Keep track of the last valid_data only
                last_valid_data = valid_data if valid_data else None

                # For memory reasons, discard training data immediately
                del train_data
                del valid_data
                gc.collect()

                
            # Compute the average loss for the whole epoch
            if total_train_samples > 0:
                train_loss = epoch_loss / total_train_samples
            else:
                train_loss = 0.0

            utils.Log.info(f'[Epoch {epoch}] Train loss: {train_loss}')

            # -- 2) Validation on the Last Chunk Only --
            self.model.eval()
            gold_labels = []
            pred_labels = []

            valid_f1 = 0.0
            if last_valid_data:
                num_valid = len(last_valid_data)
                with torch.no_grad():
                    for attmap_features, gtlabels in tqdm(last_valid_data, total=num_valid, ncols=100):
                        attmap_features = attmap_features.to(DEVICE)
                        pred_probs = self.model.get_probs(attmap_features).detach().cpu()
                        
                        gold_labels.extend(gtlabels.numpy().tolist())
                        pred_labels.extend([int(p > 0.5) for p in pred_probs.numpy().tolist()])

                valid_f1 = f1_score(gold_labels, pred_labels, average="micro")
                utils.Log.info(f'[Epoch {epoch}] Valid F1 (last part): {valid_f1:.4f}')

            # Clean up last_valid_data
            del last_valid_data
            gc.collect()

            # -- 3) Early Stopping / Best Model Logic --
            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                best_epoch = epoch
                utils.Log.info(f'[Epoch {epoch}] New best valid F1: {valid_f1:.4f}')
            else:
                utils.Log.info(f'[Epoch {epoch}] Valid F1 did not improve. Stopping early.')
                break

            # -- 4) Save Checkpoint --
            ckpt_dict = {
                'epoch': epoch,
                'model': self.model,
                'valid_f1': valid_f1,
                'train_loss': train_loss,
            }
            ckpt_path = self.output_dir / f'epoch-{epoch}.ckpt'
            torch.save(ckpt_dict, ckpt_path)
            utils.Log.info(f'[Epoch {epoch}] Saved checkpoint at {ckpt_path}')

        utils.Log.info(f"[Training Complete] Best epoch: {best_epoch}. Best valid F1: {best_valid_f1:.4f}")
        return best_epoch
