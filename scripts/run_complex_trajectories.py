#!/usr/bin/env python3
"""
Script to generate trajectories and labels for complex tasks only.

This script runs all complex tasks across all three domains (telecom, airline, retail)
and generates labeled trajectories that can be used for evaluation.

Usage:
    # Run with default settings (1 trial, gpt-4o)
    python scripts/run_complex_trajectories.py
    
    # Run with multiple trials
    python scripts/run_complex_trajectories.py --num-trials 4
    
    # Run specific domain only
    python scripts/run_complex_trajectories.py --domain telecom
    
    # Use different model
    python scripts/run_complex_trajectories.py --agent-llm gpt-4o-mini --user-llm gpt-4o-mini
    
    # Full configuration
    python scripts/run_complex_trajectories.py \
        --num-trials 4 \
        --max-steps 30 \
        --max-concurrency 10 \
        --agent-llm gpt-4o \
        --user-llm gpt-4o \
        --output-dir data/tau2/results/complex \
        --domain all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tau2.data_model.simulation import RunConfig, Results
from tau2.run import run_domain, load_tasks
from tau2.metrics.agent_metrics import avg_reward, pass_hat_k


def get_complex_task_count():
    """Get the count of complex tasks per domain."""
    counts = {}
    
    # Telecom
    telecom_tasks = load_tasks("telecom", task_split_name="complex")
    counts["telecom"] = len(telecom_tasks)
    
    # Airline
    airline_tasks = load_tasks("airline", task_split_name="complex")
    counts["airline"] = len(airline_tasks)
    
    # Retail
    retail_tasks = load_tasks("retail", task_split_name="complex")
    counts["retail"] = len(retail_tasks)
    
    return counts


def estimate_cost_and_time(task_count: int, num_trials: int, agent_llm: str):
    """Estimate cost and time for running tasks."""
    # Rough estimates based on average token usage
    # Complex tasks typically use ~5000 tokens per trajectory
    tokens_per_trajectory = 8000  # Input + output
    
    total_trajectories = task_count * num_trials
    total_tokens = total_trajectories * tokens_per_trajectory
    
    # Cost estimates (rough, varies by model)
    costs = {
        "gpt-4o": 0.005,  # per 1K tokens (blended)
        "gpt-4o-mini": 0.00015,  # per 1K tokens
        "gpt-4-turbo": 0.01,  # per 1K tokens
        "claude-3-opus": 0.015,  # per 1K tokens
        "claude-3-sonnet": 0.003,  # per 1K tokens
    }
    
    cost_per_1k = costs.get(agent_llm, 0.005)  # Default to gpt-4o
    estimated_cost = (total_tokens / 1000) * cost_per_1k * 2  # x2 for agent + user
    
    # Time estimate: ~90 seconds per trajectory with concurrency
    time_per_trajectory = 90  # seconds
    estimated_time_minutes = (total_trajectories * time_per_trajectory) / 60
    
    return {
        "total_trajectories": total_trajectories,
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 2),
        "estimated_time_minutes": round(estimated_time_minutes, 1),
    }


def run_complex_tasks(
    domains: list[str],
    num_trials: int,
    max_steps: int,
    max_concurrency: int,
    agent_llm: str,
    user_llm: str,
    output_dir: Path,
    dry_run: bool = False,
):
    """Run complex tasks for specified domains."""
    
    results = {}
    
    for domain in domains:
        print(f"\n{'='*60}")
        print(f"RUNNING {domain.upper()} COMPLEX TASKS")
        print('='*60)
        
        # Get task count
        tasks = load_tasks(domain, task_split_name="complex")
        print(f"Found {len(tasks)} complex tasks")
        
        if dry_run:
            print("(DRY RUN - not executing)")
            continue
        
        # Create config
        config = RunConfig(
            domain_name=domain,
            task_set_name=domain,
            task_split_name="complex",
            num_trials=num_trials,
            max_steps=max_steps,
            max_concurrency=max_concurrency,
            agent="llm_agent",
            agent_llm=agent_llm,
            agent_llm_args={"temperature": 0.0},
            user="user_simulator",
            user_llm=user_llm,
            user_llm_args={"temperature": 0.0},
        )
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{domain}_complex_{agent_llm}_{num_trials}trials_{timestamp}.json"
        
        print(f"Running with config:")
        print(f"  Agent LLM: {agent_llm}")
        print(f"  User LLM: {user_llm}")
        print(f"  Num trials: {num_trials}")
        print(f"  Max steps: {max_steps}")
        print(f"  Max concurrency: {max_concurrency}")
        print(f"  Output: {output_file}")
        print()
        
        # Run domain
        try:
            domain_results = run_domain(config, save_path=output_file)
            results[domain] = domain_results
            
            # Calculate metrics
            reward = avg_reward(domain_results)
            pass_1 = pass_hat_k(domain_results, k=1)
            pass_k = pass_hat_k(domain_results, k=num_trials) if num_trials > 1 else None
            
            print(f"\n{domain.upper()} RESULTS:")
            print(f"  Average Reward: {reward:.3f}")
            print(f"  Pass^1: {pass_1:.3f}")
            if pass_k is not None:
                print(f"  Pass^{num_trials}: {pass_k:.3f}")
            print(f"  Results saved to: {output_file}")
            
        except Exception as e:
            print(f"ERROR running {domain}: {e}")
            results[domain] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectories and labels for complex tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        choices=["telecom", "airline", "retail", "all"],
        help="Which domain(s) to run (default: all)"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum steps per trajectory (default: 30)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent simulations (default: 5)"
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default="gpt-4o",
        help="LLM for agent (default: gpt-4o)"
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default="gpt-4o",
        help="LLM for user simulator (default: gpt-4o)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tau2/results/complex",
        help="Output directory (default: data/tau2/results/complex)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show what would be run without executing"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only show cost/time estimates without running"
    )
    
    args = parser.parse_args()
    
    # Determine domains to run
    if args.domain == "all":
        domains = ["telecom", "airline", "retail"]
    else:
        domains = [args.domain]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Show overview
    print("="*60)
    print("COMPLEX TASK TRAJECTORY GENERATION")
    print("="*60)
    
    task_counts = get_complex_task_count()
    total_tasks = sum(task_counts[d] for d in domains)
    
    print(f"\nComplex task counts:")
    for domain, count in task_counts.items():
        marker = "✓" if domain in domains else " "
        print(f"  [{marker}] {domain}: {count} tasks")
    print(f"\nTotal tasks to run: {total_tasks}")
    print(f"Total trials per task: {args.num_trials}")
    print(f"Total trajectories: {total_tasks * args.num_trials}")
    
    # Show estimates
    estimates = estimate_cost_and_time(total_tasks, args.num_trials, args.agent_llm)
    print(f"\nEstimates:")
    print(f"  Total trajectories: {estimates['total_trajectories']}")
    print(f"  Estimated tokens: ~{estimates['estimated_tokens']:,}")
    print(f"  Estimated cost: ~${estimates['estimated_cost_usd']:.2f}")
    print(f"  Estimated time: ~{estimates['estimated_time_minutes']} minutes")
    
    if args.estimate_only:
        return
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No tasks will be executed]")
    
    # Confirm before running
    if not args.dry_run:
        print(f"\nConfiguration:")
        print(f"  Agent LLM: {args.agent_llm}")
        print(f"  User LLM: {args.user_llm}")
        print(f"  Max steps: {args.max_steps}")
        print(f"  Max concurrency: {args.max_concurrency}")
        print(f"  Output dir: {output_dir}")
        
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run tasks
    results = run_complex_tasks(
        domains=domains,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for domain, result in results.items():
        if result is None:
            print(f"{domain}: ERROR (see above)")
        else:
            reward = avg_reward(result)
            print(f"{domain}: {reward:.3f} avg reward ({len(result.simulations)} trajectories)")


if __name__ == "__main__":
    main()
