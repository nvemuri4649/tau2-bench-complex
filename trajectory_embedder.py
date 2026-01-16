#!/usr/bin/env python3
"""
Trajectory Embedding Model for JLBench

Architecture: Multi-modal Contrastive Learning
- Text Encoder: Sentence Transformer for encoding messages and assertions
- Sequence Encoder: Transformer for tool call sequences
- Fusion Layer: Combines text, sequence, and structural features
- Contrastive Training: Groups trajectories by failure mode similarity

Training: InfoNCE Loss with hard negative mining
- Positive pairs: Same failure mode (e.g., both failed same NL assertion type)
- Negative pairs: Different outcomes or different failure types
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
import hashlib

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrajectoryFeatures:
    """Extracted features from a single trajectory"""
    trajectory_id: str
    task_id: str
    domain: str
    
    # Tool call sequence
    tool_sequence: List[str]
    tool_args_hashes: List[str]  # Hashed args for similarity
    
    # Message content (truncated for efficiency)
    user_messages: List[str]
    assistant_messages: List[str]
    tool_results_summary: List[str]
    
    # Evaluation outcomes
    overall_reward: float
    nl_assertions_passed: List[bool]
    nl_assertions_text: List[str]
    env_assertions_passed: List[bool]
    communicate_passed: List[bool]
    communicate_items: List[str]
    
    # Failure mode labels (for contrastive learning)
    failure_modes: List[str]  # e.g., ["nl_failed", "communicate_failed"]
    failed_assertion_types: List[str]  # Specific assertion types that failed


@dataclass 
class ContrastivePair:
    """A pair of trajectories for contrastive learning"""
    anchor: TrajectoryFeatures
    positive: TrajectoryFeatures  # Same failure mode
    negatives: List[TrajectoryFeatures]  # Different failure modes
    similarity_label: float  # 1.0 for same, 0.0 for different


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class TrajectoryFeatureExtractor:
    """Extract structured features from raw trajectory JSON"""
    
    def __init__(self, max_messages: int = 20, max_text_len: int = 512):
        self.max_messages = max_messages
        self.max_text_len = max_text_len
    
    def extract(self, trajectory: Dict, source_file: str) -> TrajectoryFeatures:
        """Extract features from a single trajectory"""
        
        # Basic info
        task_id = trajectory.get('task_id', 'unknown')
        domain = self._extract_domain(source_file)
        
        # Tool sequence
        tool_sequence, tool_args_hashes = self._extract_tool_sequence(trajectory)
        
        # Messages
        user_msgs, assistant_msgs, tool_results = self._extract_messages(trajectory)
        
        # Evaluation outcomes
        reward_info = trajectory.get('reward_info') or {}
        
        nl_assertions = reward_info.get('nl_assertions') or []
        nl_passed = [a.get('met', False) for a in nl_assertions]
        nl_text = [a.get('assertion', '')[:200] for a in nl_assertions]
        
        env_assertions = reward_info.get('env_assertions') or []
        env_passed = [a.get('met', False) for a in env_assertions]
        
        communicate = reward_info.get('communicate_checks') or []
        comm_passed = [c.get('met', False) for c in communicate]
        comm_items = [c.get('info', '') for c in communicate]
        
        # Determine failure modes
        failure_modes = []
        failed_types = []
        
        if not all(nl_passed) and nl_passed:
            failure_modes.append('nl_failed')
            for i, (passed, text) in enumerate(zip(nl_passed, nl_text)):
                if not passed:
                    # Create a category from the assertion text
                    failed_types.append(self._categorize_nl_assertion(text))
        
        if not all(env_passed) and env_passed:
            failure_modes.append('env_failed')
            failed_types.append('env_state_mismatch')
        
        if not all(comm_passed) and comm_passed:
            failure_modes.append('communicate_failed')
            for i, (passed, item) in enumerate(zip(comm_passed, comm_items)):
                if not passed:
                    failed_types.append(f'missing_info:{item[:30]}')
        
        if not failure_modes:
            failure_modes.append('all_passed')
        
        return TrajectoryFeatures(
            trajectory_id=trajectory.get('id', hashlib.md5(str(trajectory).encode()).hexdigest()[:8]),
            task_id=task_id,
            domain=domain,
            tool_sequence=tool_sequence,
            tool_args_hashes=tool_args_hashes,
            user_messages=user_msgs,
            assistant_messages=assistant_msgs,
            tool_results_summary=tool_results,
            overall_reward=reward_info.get('reward', 0.0),
            nl_assertions_passed=nl_passed,
            nl_assertions_text=nl_text,
            env_assertions_passed=env_passed,
            communicate_passed=comm_passed,
            communicate_items=comm_items,
            failure_modes=failure_modes,
            failed_assertion_types=list(set(failed_types))
        )
    
    def _extract_domain(self, source_file: str) -> str:
        if 'telecom' in source_file.lower():
            return 'telecom'
        elif 'airline' in source_file.lower():
            return 'airline'
        elif 'retail' in source_file.lower():
            return 'retail'
        return 'unknown'
    
    def _extract_tool_sequence(self, trajectory: Dict) -> Tuple[List[str], List[str]]:
        """Extract ordered tool calls and hashed arguments"""
        tools = []
        arg_hashes = []
        
        messages = trajectory.get('messages') or []
        for m in messages:
            if isinstance(m, dict):
                tool_calls = m.get('tool_calls') or []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get('name', 'unknown')
                        args = tc.get('args', {})
                        tools.append(name)
                        # Hash args for similarity (ignoring verbose padding)
                        clean_args = {k: v for k, v in args.items() if not k.startswith('_')}
                        arg_hashes.append(hashlib.md5(str(sorted(clean_args.items())).encode()).hexdigest()[:8])
        
        return tools[:50], arg_hashes[:50]  # Limit sequence length
    
    def _extract_messages(self, trajectory: Dict) -> Tuple[List[str], List[str], List[str]]:
        """Extract user messages, assistant messages, and tool results"""
        user_msgs = []
        assistant_msgs = []
        tool_results = []
        
        messages = trajectory.get('messages') or []
        for m in messages:
            if isinstance(m, dict):
                msg_type = m.get('type') or m.get('role', '')
                content = m.get('content', '')
                
                if isinstance(content, str):
                    content = content[:self.max_text_len]
                else:
                    content = str(content)[:self.max_text_len]
                
                if msg_type == 'user' or m.get('role') == 'user':
                    user_msgs.append(content)
                elif msg_type == 'assistant' or m.get('role') == 'assistant':
                    assistant_msgs.append(content)
                elif msg_type == 'tool':
                    # Summarize tool result (remove verbose padding)
                    tool_results.append(content[:200])
        
        return (
            user_msgs[:self.max_messages],
            assistant_msgs[:self.max_messages],
            tool_results[:self.max_messages]
        )
    
    def _categorize_nl_assertion(self, assertion_text: str) -> str:
        """Categorize NL assertion into high-level types"""
        text_lower = assertion_text.lower()
        
        if 'exact' in text_lower or 'exactly' in text_lower:
            return 'exact_value_match'
        elif 'hallucin' in text_lower or 'not make up' in text_lower or 'not invent' in text_lower:
            return 'anti_hallucination'
        elif 'before' in text_lower or 'sequence' in text_lower or 'step' in text_lower:
            return 'sequential_reasoning'
        elif 'confuse' in text_lower or 'correct line' in text_lower or 'correct id' in text_lower:
            return 'entity_disambiguation'
        elif 'calculate' in text_lower or 'price' in text_lower or 'total' in text_lower:
            return 'calculation_accuracy'
        elif 'verify' in text_lower or 'check' in text_lower:
            return 'verification_required'
        else:
            return 'general_policy'


# ============================================================================
# EMBEDDING MODEL ARCHITECTURE
# ============================================================================

class ToolSequenceEncoder(nn.Module):
    """Transformer encoder for tool call sequences"""
    
    def __init__(
        self, 
        vocab_size: int = 100,  # Number of unique tools
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 50
    ):
        super().__init__()
        
        self.tool_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, tool_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tool_ids: [batch, seq_len] tool indices
            mask: [batch, seq_len] attention mask (1 = attend, 0 = ignore)
        Returns:
            [batch, embed_dim] sequence embedding
        """
        batch_size, seq_len = tool_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=tool_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.tool_embedding(tool_ids) + self.pos_embedding(positions)
        
        # Transformer
        if mask is not None:
            # Convert to attention mask format (True = ignore)
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
        
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Pool: mean over non-masked positions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return self.output_proj(x)


class TrajectoryEmbedder(nn.Module):
    """
    Multi-modal trajectory embedding model
    
    Combines:
    1. Tool sequence embeddings (Transformer)
    2. Text embeddings (from pretrained model or learned)
    3. Structural features (pass/fail signals, counts)
    """
    
    def __init__(
        self,
        tool_vocab_size: int = 100,
        tool_embed_dim: int = 128,
        text_embed_dim: int = 384,  # Sentence transformer dimension
        struct_feat_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 128,
        use_pretrained_text: bool = False
    ):
        super().__init__()
        
        # Tool sequence encoder
        self.tool_encoder = ToolSequenceEncoder(
            vocab_size=tool_vocab_size,
            embed_dim=tool_embed_dim
        )
        
        # Text encoder (simple learned embeddings if not using pretrained)
        self.use_pretrained_text = use_pretrained_text
        if not use_pretrained_text:
            # Bag-of-words style with learned embeddings
            self.text_vocab_size = 10000
            self.text_embedding = nn.EmbeddingBag(self.text_vocab_size, text_embed_dim, mode='mean')
        
        # Structural feature encoder
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Domain embedding
        self.domain_embedding = nn.Embedding(4, 32)  # 3 domains + unknown
        
        # Fusion layers
        total_input_dim = tool_embed_dim + text_embed_dim + hidden_dim // 2 + 32
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(
        self,
        tool_ids: torch.Tensor,
        tool_mask: torch.Tensor,
        text_ids: torch.Tensor,
        text_offsets: torch.Tensor,
        struct_features: torch.Tensor,
        domain_ids: torch.Tensor,
        return_projection: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            embedding: [batch, output_dim] trajectory embedding
            projection: [batch, output_dim] projection for contrastive loss
        """
        # Encode tool sequence
        tool_emb = self.tool_encoder(tool_ids, tool_mask)
        
        # Encode text
        if not self.use_pretrained_text:
            text_emb = self.text_embedding(text_ids, text_offsets)
        else:
            text_emb = text_ids  # Assume pretrained embeddings passed directly
        
        # Encode structural features
        struct_emb = self.struct_encoder(struct_features)
        
        # Domain embedding
        domain_emb = self.domain_embedding(domain_ids)
        
        # Fuse all modalities
        combined = torch.cat([tool_emb, text_emb, struct_emb, domain_emb], dim=-1)
        embedding = self.fusion(combined)
        
        # Normalize embedding
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        if return_projection:
            projection = self.projection_head(embedding)
            projection = F.normalize(projection, p=2, dim=-1)
            return embedding, projection
        
        return embedding, None


# ============================================================================
# CONTRASTIVE LEARNING
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with temperature scaling
    
    Groups trajectories by failure mode similarity:
    - Positive: Same failure mode (e.g., both failed same NL assertion type)
    - Negative: Different failure modes or different outcomes
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: [batch, dim] anchor embeddings
            positive: [batch, dim] positive embeddings  
            negatives: [batch, num_neg, dim] negative embeddings
        """
        batch_size = anchor.shape[0]
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [batch]
        
        # Negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature  # [batch, num_neg]
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [batch, 1 + num_neg]
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class FailureModeGrouper:
    """Groups trajectories by failure mode for contrastive pair generation"""
    
    def __init__(self, features_list: List[TrajectoryFeatures]):
        self.features_list = features_list
        self.groups = self._build_groups()
    
    def _build_groups(self) -> Dict[str, List[int]]:
        """Group trajectory indices by failure mode"""
        groups = defaultdict(list)
        
        for i, feat in enumerate(self.features_list):
            # Group by primary failure mode
            for mode in feat.failure_modes:
                groups[mode].append(i)
            
            # Group by specific failed assertion types
            for assert_type in feat.failed_assertion_types:
                groups[f'assert:{assert_type}'].append(i)
            
            # Group by domain
            groups[f'domain:{feat.domain}'].append(i)
            
            # Group by outcome
            if feat.overall_reward > 0.5:
                groups['outcome:passed'].append(i)
            else:
                groups['outcome:failed'].append(i)
        
        return dict(groups)
    
    def get_positive_pairs(self, anchor_idx: int) -> List[int]:
        """Get indices of trajectories similar to anchor"""
        anchor = self.features_list[anchor_idx]
        positives = set()
        
        # Find trajectories with same failure modes
        for mode in anchor.failure_modes:
            for idx in self.groups.get(mode, []):
                if idx != anchor_idx:
                    positives.add(idx)
        
        # Find trajectories with same specific failure types
        for assert_type in anchor.failed_assertion_types:
            for idx in self.groups.get(f'assert:{assert_type}', []):
                if idx != anchor_idx:
                    positives.add(idx)
        
        return list(positives)
    
    def get_negative_samples(self, anchor_idx: int, num_negatives: int = 5) -> List[int]:
        """Get indices of dissimilar trajectories"""
        anchor = self.features_list[anchor_idx]
        positives = set(self.get_positive_pairs(anchor_idx))
        positives.add(anchor_idx)
        
        # All other trajectories are potential negatives
        negatives = [i for i in range(len(self.features_list)) if i not in positives]
        
        # Prioritize hard negatives (same domain, different outcome)
        hard_negatives = []
        easy_negatives = []
        
        for idx in negatives:
            other = self.features_list[idx]
            if other.domain == anchor.domain:
                hard_negatives.append(idx)
            else:
                easy_negatives.append(idx)
        
        # Mix hard and easy negatives
        selected = []
        np.random.shuffle(hard_negatives)
        np.random.shuffle(easy_negatives)
        
        while len(selected) < num_negatives and (hard_negatives or easy_negatives):
            if hard_negatives and (not easy_negatives or np.random.random() < 0.7):
                selected.append(hard_negatives.pop())
            elif easy_negatives:
                selected.append(easy_negatives.pop())
        
        return selected


# ============================================================================
# DATASET
# ============================================================================

class TrajectoryDataset(Dataset):
    """PyTorch dataset for trajectory embeddings"""
    
    def __init__(
        self,
        features_list: List[TrajectoryFeatures],
        tool_vocab: Dict[str, int],
        text_vocab: Dict[str, int],
        domain_vocab: Dict[str, int],
        num_negatives: int = 5
    ):
        self.features_list = features_list
        self.tool_vocab = tool_vocab
        self.text_vocab = text_vocab
        self.domain_vocab = domain_vocab
        self.num_negatives = num_negatives
        
        self.grouper = FailureModeGrouper(features_list)
        
        # Pre-compute structural features
        self.struct_features = self._compute_struct_features()
    
    def _compute_struct_features(self) -> torch.Tensor:
        """Compute structural features for all trajectories"""
        features = []
        for feat in self.features_list:
            f = [
                feat.overall_reward,
                len(feat.tool_sequence) / 50.0,  # Normalized tool count
                len(feat.user_messages) / 20.0,
                len(feat.assistant_messages) / 20.0,
                sum(feat.nl_assertions_passed) / max(len(feat.nl_assertions_passed), 1),
                sum(feat.env_assertions_passed) / max(len(feat.env_assertions_passed), 1),
                sum(feat.communicate_passed) / max(len(feat.communicate_passed), 1),
                len(feat.failed_assertion_types) / 10.0,
                1.0 if 'nl_failed' in feat.failure_modes else 0.0,
                1.0 if 'env_failed' in feat.failure_modes else 0.0,
                1.0 if 'communicate_failed' in feat.failure_modes else 0.0,
                1.0 if 'all_passed' in feat.failure_modes else 0.0,
                # Assertion type indicators
                1.0 if 'exact_value_match' in feat.failed_assertion_types else 0.0,
                1.0 if 'anti_hallucination' in feat.failed_assertion_types else 0.0,
                1.0 if 'sequential_reasoning' in feat.failed_assertion_types else 0.0,
                1.0 if 'entity_disambiguation' in feat.failed_assertion_types else 0.0,
                1.0 if 'calculation_accuracy' in feat.failed_assertion_types else 0.0,
                1.0 if 'verification_required' in feat.failed_assertion_types else 0.0,
            ]
            # Pad to fixed size
            while len(f) < 32:
                f.append(0.0)
            features.append(f[:32])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _tokenize_tools(self, tools: List[str], max_len: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert tool names to indices"""
        ids = [self.tool_vocab.get(t, 0) for t in tools[:max_len]]
        mask = [1] * len(ids)
        
        # Pad
        while len(ids) < max_len:
            ids.append(0)
            mask.append(0)
        
        return torch.tensor(ids), torch.tensor(mask)
    
    def _tokenize_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple word tokenization for text"""
        all_ids = []
        for text in texts:
            words = text.lower().split()[:100]
            ids = [self.text_vocab.get(w, 0) for w in words]
            all_ids.extend(ids)
        
        if not all_ids:
            all_ids = [0]
        
        return torch.tensor(all_ids[:500]), torch.tensor([0])
    
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feat = self.features_list[idx]
        
        # Tool sequence
        tool_ids, tool_mask = self._tokenize_tools(feat.tool_sequence)
        
        # Text (combine all messages)
        all_text = feat.user_messages + feat.assistant_messages + feat.nl_assertions_text
        text_ids, text_offsets = self._tokenize_text(all_text)
        
        # Domain
        domain_id = self.domain_vocab.get(feat.domain, 0)
        
        # Get positive and negative pairs
        positives = self.grouper.get_positive_pairs(idx)
        negatives = self.grouper.get_negative_samples(idx, self.num_negatives)
        
        # Select one positive (or self if none available)
        pos_idx = positives[np.random.randint(len(positives))] if positives else idx
        
        return {
            'anchor_idx': idx,
            'positive_idx': pos_idx,
            'negative_idxs': negatives,
            'tool_ids': tool_ids,
            'tool_mask': tool_mask,
            'text_ids': text_ids,
            'text_offsets': text_offsets,
            'struct_features': self.struct_features[idx],
            'domain_id': torch.tensor(domain_id)
        }


# ============================================================================
# TRAINING
# ============================================================================

class TrajectoryEmbedderTrainer:
    """Training loop for trajectory embedding model"""
    
    def __init__(
        self,
        model: TrajectoryEmbedder,
        dataset: TrajectoryDataset,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        temperature: float = 0.07,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.loss_fn = InfoNCELoss(temperature=temperature)
        
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function"""
        result = {
            'anchor_idx': torch.tensor([b['anchor_idx'] for b in batch]),
            'positive_idx': torch.tensor([b['positive_idx'] for b in batch]),
            'tool_ids': torch.stack([b['tool_ids'] for b in batch]),
            'tool_mask': torch.stack([b['tool_mask'] for b in batch]),
            'struct_features': torch.stack([b['struct_features'] for b in batch]),
            'domain_id': torch.stack([b['domain_id'] for b in batch]),
        }
        
        # Handle variable-length text
        max_text_len = max(len(b['text_ids']) for b in batch)
        text_ids = []
        text_offsets = [0]
        current_offset = 0
        for b in batch:
            ids = b['text_ids']
            text_ids.extend(ids.tolist())
            current_offset += len(ids)
            text_offsets.append(current_offset)
        
        result['text_ids'] = torch.tensor(text_ids)
        result['text_offsets'] = torch.tensor(text_offsets[:-1])
        
        # Collect negative indices
        result['negative_idxs'] = [b['negative_idxs'] for b in batch]
        
        return result
    
    def _get_embeddings(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for a batch"""
        return self.model(
            tool_ids=batch['tool_ids'].to(self.device),
            tool_mask=batch['tool_mask'].to(self.device),
            text_ids=batch['text_ids'].to(self.device),
            text_offsets=batch['text_offsets'].to(self.device),
            struct_features=batch['struct_features'].to(self.device),
            domain_ids=batch['domain_id'].to(self.device)
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.dataloader:
            self.optimizer.zero_grad()
            
            # Get anchor embeddings
            _, anchor_proj = self._get_embeddings(batch)
            
            # Get positive embeddings
            pos_batch = self._create_batch_for_indices(batch['positive_idx'].tolist())
            _, positive_proj = self._get_embeddings(pos_batch)
            
            # Get negative embeddings
            all_neg_idxs = []
            for neg_list in batch['negative_idxs']:
                all_neg_idxs.extend(neg_list)
            
            if all_neg_idxs:
                neg_batch = self._create_batch_for_indices(all_neg_idxs)
                _, neg_proj = self._get_embeddings(neg_batch)
                
                # Reshape negatives
                batch_size = len(batch['negative_idxs'])
                num_neg = len(batch['negative_idxs'][0]) if batch['negative_idxs'] else 0
                if num_neg > 0:
                    neg_proj = neg_proj.view(batch_size, num_neg, -1)
                else:
                    neg_proj = torch.zeros(batch_size, 1, anchor_proj.shape[-1], device=self.device)
            else:
                neg_proj = torch.zeros(len(anchor_proj), 1, anchor_proj.shape[-1], device=self.device)
            
            # Compute loss
            loss = self.loss_fn(anchor_proj, positive_proj, neg_proj)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        self.scheduler.step()
        return total_loss / max(num_batches, 1)
    
    def _create_batch_for_indices(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Create a batch from trajectory indices"""
        items = [self.dataset[i] for i in indices]
        return self._collate_fn(items)
    
    def train(self, num_epochs: int = 50, log_interval: int = 10) -> List[float]:
        """Full training loop"""
        losses = []
        
        for epoch in range(num_epochs):
            loss = self.train_epoch()
            losses.append(loss)
            
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[TrajectoryFeatures]]:
        """Get embeddings for all trajectories"""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(len(self.dataset)):
                batch = self._create_batch_for_indices([i])
                emb, _ = self._get_embeddings(batch)
                embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings), self.dataset.features_list


# ============================================================================
# MAIN - BUILD AND TRAIN
# ============================================================================

def load_trajectories(data_dir: str = "data/simulations") -> List[Dict]:
    """Load all trajectory data"""
    files = [
        "telecom_complex.json", "airline_complex.json", "retail_complex.json",
        "telecom_gpt4o.json", "airline_gpt4o.json", "retail_gpt4o.json",
        "telecom_gpt52.json", "airline_gpt52.json", "retail_gpt52.json",
    ]
    
    trajectories = []
    for f in files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            with open(path) as fp:
                data = json.load(fp)
                for sim in data.get("simulations", []):
                    sim["_source_file"] = f
                    trajectories.append(sim)
    
    return trajectories


def build_vocabularies(features_list: List[TrajectoryFeatures]) -> Tuple[Dict, Dict, Dict]:
    """Build vocabularies for tools, text, and domains"""
    # Tool vocab
    tools = set()
    for feat in features_list:
        tools.update(feat.tool_sequence)
    tool_vocab = {t: i+1 for i, t in enumerate(sorted(tools))}  # 0 = padding
    
    # Text vocab (simple word-level)
    words = set()
    for feat in features_list:
        for text in feat.user_messages + feat.assistant_messages + feat.nl_assertions_text:
            words.update(text.lower().split())
    # Limit vocab size
    word_counts = defaultdict(int)
    for feat in features_list:
        for text in feat.user_messages + feat.assistant_messages:
            for w in text.lower().split():
                word_counts[w] += 1
    top_words = sorted(word_counts.keys(), key=lambda w: word_counts[w], reverse=True)[:9999]
    text_vocab = {w: i+1 for i, w in enumerate(top_words)}
    
    # Domain vocab
    domain_vocab = {'unknown': 0, 'telecom': 1, 'airline': 2, 'retail': 3}
    
    return tool_vocab, text_vocab, domain_vocab


def main():
    """Main training script"""
    print("="*60)
    print("TRAJECTORY EMBEDDING MODEL - TRAINING")
    print("="*60)
    
    # Load data
    print("\n1. Loading trajectories...")
    trajectories = load_trajectories()
    print(f"   Loaded {len(trajectories)} trajectories")
    
    # Extract features
    print("\n2. Extracting features...")
    extractor = TrajectoryFeatureExtractor()
    features_list = [extractor.extract(t, t.get('_source_file', '')) for t in trajectories]
    print(f"   Extracted features for {len(features_list)} trajectories")
    
    # Analyze failure modes
    print("\n3. Failure mode distribution:")
    mode_counts = defaultdict(int)
    for feat in features_list:
        for mode in feat.failure_modes:
            mode_counts[mode] += 1
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"   {mode}: {count}")
    
    # Build vocabularies
    print("\n4. Building vocabularies...")
    tool_vocab, text_vocab, domain_vocab = build_vocabularies(features_list)
    print(f"   Tool vocab size: {len(tool_vocab)}")
    print(f"   Text vocab size: {len(text_vocab)}")
    
    # Create dataset
    print("\n5. Creating dataset...")
    dataset = TrajectoryDataset(
        features_list=features_list,
        tool_vocab=tool_vocab,
        text_vocab=text_vocab,
        domain_vocab=domain_vocab,
        num_negatives=5
    )
    
    # Create model
    print("\n6. Creating model...")
    model = TrajectoryEmbedder(
        tool_vocab_size=len(tool_vocab) + 1,
        tool_embed_dim=128,
        text_embed_dim=384,
        struct_feat_dim=32,
        hidden_dim=256,
        output_dim=128
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Train
    print("\n7. Training...")
    trainer = TrajectoryEmbedderTrainer(
        model=model,
        dataset=dataset,
        learning_rate=1e-4,
        batch_size=8,
        temperature=0.07,
        device='cpu'  # Use 'cuda' if available
    )
    
    losses = trainer.train(num_epochs=50, log_interval=10)
    
    # Get final embeddings
    print("\n8. Generating final embeddings...")
    embeddings, features = trainer.get_all_embeddings()
    print(f"   Embedding shape: {embeddings.shape}")
    
    # Save results
    print("\n9. Saving model and embeddings...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tool_vocab': tool_vocab,
        'text_vocab': text_vocab,
        'domain_vocab': domain_vocab,
        'embeddings': embeddings,
        'training_losses': losses,
    }, 'trajectory_embedder.pt')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Model saved to: trajectory_embedder.pt")
    
    return model, embeddings, features


if __name__ == "__main__":
    main()
