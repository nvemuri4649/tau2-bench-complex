import json
import re
from typing import List, Dict, Any

from loguru import logger

from tau2.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_LLM_NL_ASSERTIONS_ARGS
from tau2.data_model.message import Message, SystemMessage, UserMessage
from tau2.data_model.simulation import NLAssertionCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.utils.llm_utils import generate

# Chunking configuration for long trajectories
CHUNK_SIZE_TOKENS = 50000  # 50k tokens per chunk
CHUNK_OVERLAP_TOKENS = 25000  # 25k token overlap between chunks
SINGLE_CONTEXT_THRESHOLD = 80000  # Use single-pass if under 80k tokens


def _strip_verbose_padding_from_content(content: str) -> str:
    """Strip verbose padding keys from tool response JSON to reduce token count."""
    if not content or not content.strip().startswith("{"):
        return content
    
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            # Remove all keys starting with underscore (verbose padding)
            keys_to_remove = [k for k in data.keys() if k.startswith("_")]
            for key in keys_to_remove:
                del data[key]
            return json.dumps(data, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass
    return content


def _clean_trajectory_messages(trajectory: list[Message]) -> list[str]:
    """Clean trajectory messages by stripping verbose padding from tool responses."""
    cleaned_messages = []
    for i, msg in enumerate(trajectory):
        content = msg.content or ""
        if msg.role == "tool":
            content = _strip_verbose_padding_from_content(content)
        # Add message index for ordering context
        cleaned_messages.append(f"[MSG {i}] {msg.role}: {content}")
    return cleaned_messages


def _chunk_trajectory(trajectory_lines: list[str], chunk_size_chars: int, overlap_chars: int) -> list[tuple[str, int, int]]:
    """
    Split trajectory into overlapping chunks.
    
    Returns:
        List of tuples: (chunk_text, start_msg_index, end_msg_index)
    """
    full_text = "\n".join(trajectory_lines)
    
    if len(full_text) <= chunk_size_chars:
        return [(full_text, 0, len(trajectory_lines) - 1)]
    
    chunks = []
    start_pos = 0
    chunk_num = 0
    
    while start_pos < len(full_text):
        end_pos = min(start_pos + chunk_size_chars, len(full_text))
        
        # Try to end at a message boundary
        if end_pos < len(full_text):
            # Look for [MSG X] pattern to find good break point
            search_region = full_text[max(end_pos - 1000, start_pos):end_pos]
            msg_matches = list(re.finditer(r'\n\[MSG \d+\]', search_region))
            if msg_matches:
                # Adjust end_pos to break at message boundary
                last_match = msg_matches[-1]
                adjusted_end = max(end_pos - 1000, start_pos) + last_match.start()
                if adjusted_end > start_pos:
                    end_pos = adjusted_end
        
        chunk_text = full_text[start_pos:end_pos]
        
        # Extract message indices from chunk
        msg_indices = [int(m.group(1)) for m in re.finditer(r'\[MSG (\d+)\]', chunk_text)]
        start_idx = min(msg_indices) if msg_indices else 0
        end_idx = max(msg_indices) if msg_indices else len(trajectory_lines) - 1
        
        chunks.append((chunk_text, start_idx, end_idx))
        chunk_num += 1
        
        # Move start position with overlap
        start_pos = end_pos - overlap_chars
        if start_pos >= len(full_text) - overlap_chars:
            break
    
    return chunks


def _extract_evidence_from_chunk(
    chunk_text: str, 
    chunk_num: int, 
    total_chunks: int,
    start_msg: int,
    end_msg: int,
    nl_assertions: list[str]
) -> Dict[str, Any]:
    """
    MAP PHASE: Extract evidence relevant to NL assertions from a single chunk.
    """
    system_prompt = """You are an evidence extractor. Given a chunk of a conversation transcript and a list of assertions to evaluate, extract any relevant evidence from this chunk.

For each assertion:
1. Look for any statements, actions, or events in this chunk that are relevant to evaluating the assertion
2. Note the message numbers [MSG X] where you find evidence
3. If this chunk contains no relevant evidence for an assertion, say "No relevant evidence in this chunk"

Be thorough but concise. Quote specific relevant text when helpful.

OUTPUT FORMAT (JSON):
{
    "chunk_info": {
        "chunk_number": <number>,
        "message_range": "<start>-<end>"
    },
    "evidence": [
        {
            "assertion": "<the assertion>",
            "relevant_evidence": "<extracted evidence with message numbers>",
            "evidence_type": "supports" | "contradicts" | "partial" | "none"
        }
    ]
}"""

    user_prompt = f"""CHUNK {chunk_num + 1} OF {total_chunks} (Messages {start_msg}-{end_msg}):

{chunk_text}

---

ASSERTIONS TO FIND EVIDENCE FOR:
{json.dumps(nl_assertions, indent=2)}

Extract relevant evidence from this chunk for each assertion."""

    messages = [
        SystemMessage(role="system", content=system_prompt),
        UserMessage(role="user", content=user_prompt),
    ]

    try:
        response = generate(
            model=DEFAULT_LLM_NL_ASSERTIONS,
            messages=messages,
            **DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        )
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"Error extracting evidence from chunk {chunk_num}: {e}")
        return {
            "chunk_info": {"chunk_number": chunk_num, "message_range": f"{start_msg}-{end_msg}"},
            "evidence": [{"assertion": a, "relevant_evidence": "Error extracting", "evidence_type": "none"} for a in nl_assertions],
            "error": str(e)
        }


def _reduce_evidence_and_judge(
    all_evidence: list[Dict[str, Any]],
    nl_assertions: list[str],
    total_messages: int
) -> list[NLAssertionCheck]:
    """
    REDUCE PHASE: Combine evidence from all chunks and make final judgment.
    """
    # Organize evidence by assertion
    evidence_by_assertion = {a: [] for a in nl_assertions}
    
    for chunk_evidence in all_evidence:
        chunk_info = chunk_evidence.get("chunk_info", {})
        for ev in chunk_evidence.get("evidence", []):
            assertion = ev.get("assertion", "")
            # Find matching assertion (may not be exact match due to LLM rephrasing)
            for original_assertion in nl_assertions:
                if original_assertion.lower()[:50] in assertion.lower() or assertion.lower()[:50] in original_assertion.lower():
                    evidence_by_assertion[original_assertion].append({
                        "chunk": chunk_info.get("chunk_number", "?"),
                        "messages": chunk_info.get("message_range", "?"),
                        "evidence": ev.get("relevant_evidence", ""),
                        "type": ev.get("evidence_type", "none")
                    })
                    break

    # Build summary for final judgment
    evidence_summary = []
    for assertion in nl_assertions:
        evidence_list = evidence_by_assertion[assertion]
        summary = f"\n### Assertion: {assertion}\n"
        if evidence_list:
            for ev in evidence_list:
                if ev["type"] != "none":
                    summary += f"- Chunk {ev['chunk']} (msgs {ev['messages']}): [{ev['type'].upper()}] {ev['evidence']}\n"
        else:
            summary += "- No evidence found across any chunks\n"
        evidence_summary.append(summary)

    system_prompt = """You are the final judge evaluating whether a conversation trajectory meets certain assertions.

You will be given:
1. A list of assertions to evaluate
2. Evidence extracted from different chunks of the conversation (the conversation was split into chunks for processing)

Based on the evidence, determine if each assertion was MET or NOT MET.

IMPORTANT:
- Consider the chronological order (message numbers indicate sequence)
- "supports" evidence means the assertion appears to be met
- "contradicts" evidence means the assertion was violated
- "partial" evidence means incomplete support
- If an assertion requires X to happen BEFORE Y, check the message numbers

OUTPUT FORMAT (JSON):
{
    "results": [
        {
            "expectedOutcome": "<the assertion>",
            "reasoning": "<explanation based on the evidence>",
            "metExpectation": true | false
        }
    ]
}"""

    user_prompt = f"""CONVERSATION INFO:
- Total messages: {total_messages}
- Processed in chunks with overlapping windows

EVIDENCE SUMMARY:
{"".join(evidence_summary)}

---

ASSERTIONS TO JUDGE:
{json.dumps(nl_assertions, indent=2)}

Based on the evidence collected from all chunks, determine if each assertion was met."""

    messages = [
        SystemMessage(role="system", content=system_prompt),
        UserMessage(role="user", content=user_prompt),
    ]

    try:
        response = generate(
            model=DEFAULT_LLM_NL_ASSERTIONS,
            messages=messages,
            **DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        )
        result_data = json.loads(response.content)
        return [
            NLAssertionCheck(
                nl_assertion=result["expectedOutcome"],
                met=result["metExpectation"],
                justification=result["reasoning"],
            )
            for result in result_data.get("results", [])
        ]
    except Exception as e:
        logger.error(f"Error in reduce phase: {e}")
        # Return all assertions as not met if reduce fails
        return [
            NLAssertionCheck(
                nl_assertion=a,
                met=False,
                justification=f"Error in evaluation: {str(e)}"
            )
            for a in nl_assertions
        ]


class NLAssertionsEvaluator:
    """
    Judge that evaluates whether a trajectory adheres to all the natural-language assertions.
    
    For long trajectories (>80k tokens), uses a Map-Reduce approach:
    1. MAP: Split trajectory into overlapping chunks, extract evidence from each
    2. REDUCE: Combine evidence and make final judgment
    
    For shorter trajectories, uses single-pass evaluation.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation by using an LLM to evaluate whether the trajectory adheres to all the natural-language assertions
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )
        nl_assertions = task.evaluation_criteria.nl_assertions
        if not nl_assertions:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No nl_assertions to evaluate"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        nl_assertions_checks = cls.evaluate_nl_assertions(
            full_trajectory, nl_assertions
        )

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.met for result in nl_assertions_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            nl_assertions=nl_assertions_checks,
            reward_breakdown={RewardType.NL_ASSERTION: reward},
        )

    @classmethod
    def evaluate_nl_assertions(
        cls,
        trajectory: list[Message],
        nl_assertions: list[str],
    ) -> list[NLAssertionCheck]:
        """
        Evaluate whether the trajectory meets each expected outcome.
        
        For long trajectories, uses chunked Map-Reduce approach.
        For short trajectories, uses single-pass evaluation.
        """
        # Clean and prepare trajectory
        cleaned_messages = _clean_trajectory_messages(trajectory)
        full_trajectory_str = "\n".join(cleaned_messages)
        estimated_tokens = len(full_trajectory_str) // 4
        
        logger.info(f"NL Evaluation: {estimated_tokens} estimated tokens, {len(trajectory)} messages")
        
        # Decide evaluation strategy based on size
        if estimated_tokens <= SINGLE_CONTEXT_THRESHOLD:
            logger.info("Using single-pass NL evaluation")
            return cls._evaluate_single_pass(full_trajectory_str, nl_assertions)
        else:
            logger.info(f"Using chunked Map-Reduce NL evaluation ({estimated_tokens} tokens > {SINGLE_CONTEXT_THRESHOLD} threshold)")
            return cls._evaluate_chunked(cleaned_messages, nl_assertions, len(trajectory))

    @classmethod
    def _evaluate_single_pass(
        cls,
        trajectory_str: str,
        nl_assertions: list[str],
    ) -> list[NLAssertionCheck]:
        """Original single-pass evaluation for shorter trajectories."""
        system_prompt = """
        TASK
        - You will be given a list of expected outcomes and a conversation that was collected during a test case run.
        - The conversation is between an agent and a customer.
        - Your job is to evaluate whether the agent satisfies each of the expected outcomes.
        - Grade each expected outcome individually.

        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your classification
        - `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
        - `expectedOutcome`: repeat the expectation from the input that you are grading
        
        Example response structure:
        {
            "results": [
                {
                    "expectedOutcome": "<one of the expected outcomes from the input>",
                    "reasoning": "<reasoning trace>",
                    "metExpectation": <false or true>,
                }
            ]
        }
        """

        user_prompt = f"""
        conversation:
        {trajectory_str}
        
        expectedOutcomes:
        {nl_assertions}
        """

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=DEFAULT_LLM_NL_ASSERTIONS,
            messages=messages,
            **DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        )
        result_data = json.loads(assistant_message.content)
        return [
            NLAssertionCheck(
                nl_assertion=result["expectedOutcome"],
                met=result["metExpectation"],
                justification=result["reasoning"],
            )
            for result in result_data.get("results", [])
        ]

    @classmethod
    def _evaluate_chunked(
        cls,
        cleaned_messages: list[str],
        nl_assertions: list[str],
        total_messages: int,
    ) -> list[NLAssertionCheck]:
        """
        Map-Reduce evaluation for long trajectories.
        
        1. MAP: Split into chunks, extract evidence from each
        2. REDUCE: Combine evidence and judge
        """
        # Convert token sizes to char sizes (4 chars per token)
        chunk_size_chars = CHUNK_SIZE_TOKENS * 4
        overlap_chars = CHUNK_OVERLAP_TOKENS * 4
        
        # Create chunks
        chunks = _chunk_trajectory(cleaned_messages, chunk_size_chars, overlap_chars)
        logger.info(f"Split trajectory into {len(chunks)} chunks for evaluation")
        
        # MAP PHASE: Extract evidence from each chunk
        all_evidence = []
        for i, (chunk_text, start_msg, end_msg) in enumerate(chunks):
            logger.debug(f"Extracting evidence from chunk {i+1}/{len(chunks)} (msgs {start_msg}-{end_msg})")
            evidence = _extract_evidence_from_chunk(
                chunk_text, i, len(chunks), start_msg, end_msg, nl_assertions
            )
            all_evidence.append(evidence)
        
        # REDUCE PHASE: Combine evidence and make final judgment
        logger.info("Reducing evidence and making final judgment")
        return _reduce_evidence_and_judge(all_evidence, nl_assertions, total_messages)
