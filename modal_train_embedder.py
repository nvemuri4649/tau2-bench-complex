#!/usr/bin/env python3
"""
Modal script for training trajectory embeddings on cloud GPU.

Usage:
    modal run modal_train_embedder.py

This will:
1. Upload trajectory data to Modal
2. Train the embedding model on GPU
3. Download embeddings and visualizations
"""

import modal

# Create Modal app
app = modal.App("trajectory-embedder")

# Define the image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "umap-learn",
)


@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU (cheap and fast enough)
    timeout=600,  # 10 min timeout
)
def train_embeddings(trajectory_data: dict):
    """Train trajectory embedding model on GPU
    
    Args:
        trajectory_data: Dict mapping filename to list of trajectory dicts
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import json
    import os
    from collections import defaultdict
    import hashlib
    from dataclasses import dataclass
    from typing import List, Dict, Tuple, Optional
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ========================================================================
    # DATA STRUCTURES
    # ========================================================================
    
    @dataclass
    class TrajectoryFeatures:
        trajectory_id: str
        task_id: str
        domain: str
        tool_sequence: List[str]
        user_messages: List[str]
        assistant_messages: List[str]
        overall_reward: float
        nl_assertions_passed: List[bool]
        nl_assertions_text: List[str]
        env_assertions_passed: List[bool]
        communicate_passed: List[bool]
        failure_modes: List[str]
        failed_assertion_types: List[str]
    
    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================
    
    def extract_domain(source_file: str) -> str:
        if 'telecom' in source_file.lower():
            return 'telecom'
        elif 'airline' in source_file.lower():
            return 'airline'
        elif 'retail' in source_file.lower():
            return 'retail'
        return 'unknown'
    
    def categorize_nl_assertion(text: str) -> str:
        text_lower = text.lower()
        if 'exact' in text_lower:
            return 'exact_value_match'
        elif 'hallucin' in text_lower or 'not make up' in text_lower:
            return 'anti_hallucination'
        elif 'before' in text_lower or 'sequence' in text_lower:
            return 'sequential_reasoning'
        elif 'confuse' in text_lower:
            return 'entity_disambiguation'
        elif 'calculate' in text_lower or 'price' in text_lower:
            return 'calculation_accuracy'
        elif 'verify' in text_lower:
            return 'verification_required'
        return 'general_policy'
    
    def extract_features(trajectory: Dict, source_file: str) -> TrajectoryFeatures:
        task_id = trajectory.get('task_id', 'unknown')
        domain = extract_domain(source_file)
        
        # Tool sequence
        tools = []
        messages = trajectory.get('messages') or []
        user_msgs, assistant_msgs = [], []
        
        for m in messages:
            if isinstance(m, dict):
                tool_calls = m.get('tool_calls') or []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tools.append(tc.get('name', 'unknown'))
                
                content = str(m.get('content', ''))[:500]
                msg_type = m.get('type') or m.get('role', '')
                if msg_type == 'user':
                    user_msgs.append(content)
                elif msg_type == 'assistant':
                    assistant_msgs.append(content)
        
        # Evaluation outcomes
        reward_info = trajectory.get('reward_info') or {}
        
        nl_assertions = reward_info.get('nl_assertions') or []
        nl_passed = [a.get('met', False) for a in nl_assertions]
        nl_text = [a.get('assertion', '')[:200] for a in nl_assertions]
        
        env_assertions = reward_info.get('env_assertions') or []
        env_passed = [a.get('met', False) for a in env_assertions]
        
        communicate = reward_info.get('communicate_checks') or []
        comm_passed = [c.get('met', False) for c in communicate]
        
        # Failure modes
        failure_modes = []
        failed_types = []
        
        if nl_passed and not all(nl_passed):
            failure_modes.append('nl_failed')
            for passed, text in zip(nl_passed, nl_text):
                if not passed:
                    failed_types.append(categorize_nl_assertion(text))
        
        if env_passed and not all(env_passed):
            failure_modes.append('env_failed')
        
        if comm_passed and not all(comm_passed):
            failure_modes.append('communicate_failed')
        
        if not failure_modes:
            failure_modes.append('all_passed')
        
        return TrajectoryFeatures(
            trajectory_id=trajectory.get('id', '')[:8],
            task_id=task_id,
            domain=domain,
            tool_sequence=tools[:50],
            user_messages=user_msgs[:20],
            assistant_messages=assistant_msgs[:20],
            overall_reward=reward_info.get('reward', 0.0),
            nl_assertions_passed=nl_passed,
            nl_assertions_text=nl_text,
            env_assertions_passed=env_passed,
            communicate_passed=comm_passed,
            failure_modes=failure_modes,
            failed_assertion_types=list(set(failed_types))
        )
    
    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
    
    class ToolSequenceEncoder(nn.Module):
        def __init__(self, vocab_size=100, embed_dim=128, num_heads=4, num_layers=2):
            super().__init__()
            self.tool_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_embedding = nn.Embedding(50, embed_dim)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, 
                dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        def forward(self, tool_ids, mask=None):
            B, L = tool_ids.shape
            pos = torch.arange(L, device=tool_ids.device).unsqueeze(0).expand(B, -1)
            x = self.tool_embedding(tool_ids) + self.pos_embedding(pos)
            
            attn_mask = ~mask.bool() if mask is not None else None
            x = self.transformer(x, src_key_padding_mask=attn_mask)
            
            if mask is not None:
                mask_exp = mask.unsqueeze(-1).float()
                x = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            
            return self.output_proj(x)
    
    class TrajectoryEmbedder(nn.Module):
        def __init__(self, tool_vocab_size=100, output_dim=128):
            super().__init__()
            self.tool_encoder = ToolSequenceEncoder(vocab_size=tool_vocab_size)
            self.text_embedding = nn.EmbeddingBag(10000, 384, mode='mean')
            self.struct_encoder = nn.Sequential(
                nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128)
            )
            self.domain_embedding = nn.Embedding(4, 32)
            
            self.fusion = nn.Sequential(
                nn.Linear(128 + 384 + 128 + 32, 256),
                nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
                nn.Linear(256, output_dim)
            )
            self.projection = nn.Sequential(
                nn.Linear(output_dim, output_dim), nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
        
        def forward(self, tool_ids, tool_mask, text_ids, text_offsets, struct_feat, domain_ids):
            tool_emb = self.tool_encoder(tool_ids, tool_mask)
            text_emb = self.text_embedding(text_ids, text_offsets)
            struct_emb = self.struct_encoder(struct_feat)
            domain_emb = self.domain_embedding(domain_ids)
            
            combined = torch.cat([tool_emb, text_emb, struct_emb, domain_emb], dim=-1)
            embedding = F.normalize(self.fusion(combined), p=2, dim=-1)
            projection = F.normalize(self.projection(embedding), p=2, dim=-1)
            
            return embedding, projection
    
    # ========================================================================
    # LOAD DATA (passed from local entrypoint)
    # ========================================================================
    
    print("Processing trajectories...")
    trajectories = []
    for filename, data in trajectory_data.items():
        for sim in data.get("simulations", []):
            sim["_source_file"] = filename
            trajectories.append(sim)
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Extract features
    print("Extracting features...")
    features_list = [extract_features(t, t.get('_source_file', '')) for t in trajectories]
    
    # Build vocabularies
    tools = set()
    words = set()
    for feat in features_list:
        tools.update(feat.tool_sequence)
        for text in feat.user_messages + feat.assistant_messages:
            words.update(text.lower().split())
    
    tool_vocab = {t: i+1 for i, t in enumerate(sorted(tools))}
    text_vocab = {w: i+1 for i, w in enumerate(list(words)[:9999])}
    domain_vocab = {'unknown': 0, 'telecom': 1, 'airline': 2, 'retail': 3}
    
    print(f"Tool vocab: {len(tool_vocab)}, Text vocab: {len(text_vocab)}")
    
    # ========================================================================
    # PREPARE DATA
    # ========================================================================
    
    def prepare_batch(features_list, indices, tool_vocab, text_vocab, domain_vocab):
        tool_ids_list, tool_masks = [], []
        text_ids_all, text_offsets = [], [0]
        struct_features = []
        domain_ids = []
        
        for idx in indices:
            feat = features_list[idx]
            
            # Tools
            ids = [tool_vocab.get(t, 0) for t in feat.tool_sequence[:50]]
            mask = [1] * len(ids)
            while len(ids) < 50:
                ids.append(0)
                mask.append(0)
            tool_ids_list.append(ids)
            tool_masks.append(mask)
            
            # Text
            all_text = feat.user_messages + feat.assistant_messages
            for text in all_text:
                for w in text.lower().split()[:50]:
                    if w in text_vocab:
                        text_ids_all.append(text_vocab[w])
            text_offsets.append(len(text_ids_all))
            
            # Structural
            sf = [
                feat.overall_reward,
                len(feat.tool_sequence) / 50.0,
                sum(feat.nl_assertions_passed) / max(len(feat.nl_assertions_passed), 1),
                sum(feat.env_assertions_passed) / max(len(feat.env_assertions_passed), 1),
                sum(feat.communicate_passed) / max(len(feat.communicate_passed), 1),
                1.0 if 'nl_failed' in feat.failure_modes else 0.0,
                1.0 if 'env_failed' in feat.failure_modes else 0.0,
                1.0 if 'communicate_failed' in feat.failure_modes else 0.0,
                1.0 if 'all_passed' in feat.failure_modes else 0.0,
            ]
            while len(sf) < 32:
                sf.append(0.0)
            struct_features.append(sf[:32])
            
            # Domain
            domain_ids.append(domain_vocab.get(feat.domain, 0))
        
        if not text_ids_all:
            text_ids_all = [0]
        
        return {
            'tool_ids': torch.tensor(tool_ids_list, device=device),
            'tool_mask': torch.tensor(tool_masks, device=device),
            'text_ids': torch.tensor(text_ids_all, device=device),
            'text_offsets': torch.tensor(text_offsets[:-1], device=device),
            'struct_feat': torch.tensor(struct_features, dtype=torch.float32, device=device),
            'domain_ids': torch.tensor(domain_ids, device=device),
        }
    
    # Group by failure mode for contrastive pairs
    mode_groups = defaultdict(list)
    for i, feat in enumerate(features_list):
        for mode in feat.failure_modes:
            mode_groups[mode].append(i)
    
    # ========================================================================
    # TRAIN
    # ========================================================================
    
    print("Creating model...")
    model = TrajectoryEmbedder(tool_vocab_size=len(tool_vocab)+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training...")
    
    num_epochs = 100
    batch_size = 16
    temperature = 0.07
    
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle indices
        indices = list(range(len(features_list)))
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < 2:
                continue
            
            optimizer.zero_grad()
            
            # Get anchor embeddings
            batch = prepare_batch(features_list, batch_indices, tool_vocab, text_vocab, domain_vocab)
            _, projections = model(**batch)
            
            # Simple contrastive: use batch as negatives
            # Positive = same failure mode, Negative = different
            sim_matrix = torch.mm(projections, projections.t()) / temperature
            
            # Create labels based on failure mode similarity
            labels = torch.zeros(len(batch_indices), len(batch_indices), device=device)
            for bi, idx_i in enumerate(batch_indices):
                for bj, idx_j in enumerate(batch_indices):
                    if bi != bj:
                        modes_i = set(features_list[idx_i].failure_modes)
                        modes_j = set(features_list[idx_j].failure_modes)
                        if modes_i & modes_j:  # Shared failure mode
                            labels[bi, bj] = 1.0
            
            # InfoNCE-style loss
            # Mask out diagonal
            mask = torch.eye(len(batch_indices), device=device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
            
            # For each anchor, compute loss
            loss = 0.0
            for bi in range(len(batch_indices)):
                positives = labels[bi].nonzero().squeeze(-1)
                if len(positives) > 0:
                    pos_sim = sim_matrix[bi, positives].logsumexp(dim=0)
                    all_sim = sim_matrix[bi].logsumexp(dim=0)
                    loss += -pos_sim + all_sim
            
            loss = loss / len(batch_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # ========================================================================
    # GENERATE FINAL EMBEDDINGS
    # ========================================================================
    
    print("Generating final embeddings...")
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(features_list), batch_size):
            batch_indices = list(range(i, min(i+batch_size, len(features_list))))
            batch = prepare_batch(features_list, batch_indices, tool_vocab, text_vocab, domain_vocab)
            embeddings, _ = model(**batch)
            all_embeddings.append(embeddings.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    
    # ========================================================================
    # VISUALIZE WITH UMAP
    # ========================================================================
    
    print("Creating UMAP visualization...")
    from umap import UMAP
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Color by failure mode
    colors = []
    labels = []
    for feat in features_list:
        if 'all_passed' in feat.failure_modes:
            colors.append('#059669')  # Green
            labels.append('Passed')
        elif 'nl_failed' in feat.failure_modes:
            colors.append('#dc2626')  # Red
            labels.append('NL Failed')
        elif 'communicate_failed' in feat.failure_modes:
            colors.append('#f59e0b')  # Orange
            labels.append('Communicate Failed')
        else:
            colors.append('#6366f1')  # Purple
            labels.append('Other Failed')
    
    plt.figure(figsize=(12, 10))
    
    # Plot by category
    for label_type in ['Passed', 'NL Failed', 'Communicate Failed', 'Other Failed']:
        mask = [l == label_type for l in labels]
        if any(mask):
            x = embeddings_2d[mask, 0]
            y = embeddings_2d[mask, 1]
            c = [colors[i] for i, m in enumerate(mask) if m]
            plt.scatter(x, y, c=c, label=label_type, alpha=0.7, s=80, edgecolors='white', linewidths=0.5)
    
    plt.legend(fontsize=12)
    plt.title('Trajectory Embeddings by Failure Mode', fontsize=16, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.tight_layout()
    
    # Save to bytes
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plot_bytes = buf.read()
    plt.close()
    
    # Color by domain
    plt.figure(figsize=(12, 10))
    domain_colors = {'telecom': '#0D7377', 'airline': '#4A90A4', 'retail': '#2E8B57', 'unknown': '#888888'}
    
    for domain in ['telecom', 'airline', 'retail']:
        mask = [f.domain == domain for f in features_list]
        if any(mask):
            x = embeddings_2d[mask, 0]
            y = embeddings_2d[mask, 1]
            plt.scatter(x, y, c=domain_colors[domain], label=domain.capitalize(), 
                       alpha=0.7, s=80, edgecolors='white', linewidths=0.5)
    
    plt.legend(fontsize=12)
    plt.title('Trajectory Embeddings by Domain', fontsize=16, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.tight_layout()
    
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf2.seek(0)
    plot_bytes_domain = buf2.read()
    plt.close()
    
    # ========================================================================
    # RETURN RESULTS
    # ========================================================================
    
    print("Done!")
    
    return {
        'embeddings': embeddings.tolist(),
        'embeddings_2d': embeddings_2d.tolist(),
        'features': [
            {
                'trajectory_id': f.trajectory_id,
                'task_id': f.task_id,
                'domain': f.domain,
                'failure_modes': f.failure_modes,
                'failed_assertion_types': f.failed_assertion_types,
                'overall_reward': f.overall_reward,
            }
            for f in features_list
        ],
        'training_losses': losses,
        'plot_failure_mode': plot_bytes,
        'plot_domain': plot_bytes_domain,
    }


@app.local_entrypoint()
def main():
    """Run training and save results locally"""
    import json
    import os
    
    # Load data locally
    print("Loading trajectory data locally...")
    data_dir = "data/simulations"
    files = [
        "telecom_complex.json", "airline_complex.json", "retail_complex.json",
        "telecom_gpt4o.json", "airline_gpt4o.json", "retail_gpt4o.json",
        "telecom_gpt52.json", "airline_gpt52.json", "retail_gpt52.json",
    ]
    
    trajectory_data = {}
    for f in files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            with open(path) as fp:
                trajectory_data[f] = json.load(fp)
            print(f"  Loaded {f}")
    
    print(f"Loaded {len(trajectory_data)} files")
    
    print("\nStarting Modal training job...")
    result = train_embeddings.remote(trajectory_data)
    
    # Save embeddings
    print("Saving embeddings...")
    with open('trajectory_embeddings.json', 'w') as f:
        json.dump({
            'embeddings': result['embeddings'],
            'embeddings_2d': result['embeddings_2d'],
            'features': result['features'],
            'training_losses': result['training_losses'],
        }, f)
    
    # Save plots
    print("Saving visualizations...")
    with open('figs/trajectory_embeddings_failure_mode.png', 'wb') as f:
        f.write(result['plot_failure_mode'])
    
    with open('figs/trajectory_embeddings_domain.png', 'wb') as f:
        f.write(result['plot_domain'])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final loss: {result['training_losses'][-1]:.4f}")
    print(f"Embeddings saved to: trajectory_embeddings.json")
    print(f"Plots saved to: figs/trajectory_embeddings_*.png")


if __name__ == "__main__":
    # For local testing without Modal
    print("Run with: modal run modal_train_embedder.py")
