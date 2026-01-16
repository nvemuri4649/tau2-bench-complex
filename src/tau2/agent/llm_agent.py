from copy import deepcopy
from typing import List, Optional
import json

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.data_model.tasks import Action, Task
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate

# Context window limits for different models (in tokens)
MODEL_CONTEXT_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1": 128000,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 16000,
    "gpt-5.2": 400000,
    "claude-3-5-sonnet": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-opus-4-5": 200000,
}
DEFAULT_CONTEXT_LIMIT = 128000
CONTEXT_BUFFER = 15000  # Leave buffer for response and overhead


def _estimate_tokens(messages: list, tools: list = None) -> int:
    """Estimate token count for messages and tools.
    
    Uses conservative 3 chars per token ratio to avoid underestimation.
    Also adds overhead for JSON structure and message metadata.
    """
    total_chars = 0
    for msg in messages:
        content = getattr(msg, 'content', '') or ''
        total_chars += len(content)
        # Add overhead for message structure (role, etc.)
        total_chars += 50
        # Account for tool calls in assistant messages
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    tc_str = json.dumps(tc.model_dump() if hasattr(tc, 'model_dump') else str(tc))
                    total_chars += len(tc_str)
                except:
                    total_chars += 500  # Fallback estimate
    
    # Estimate tools definition size
    if tools:
        for tool in tools:
            try:
                tool_str = str(tool.to_json_schema() if hasattr(tool, 'to_json_schema') else str(tool))
                total_chars += len(tool_str)
            except:
                total_chars += 1000  # Fallback estimate per tool
    
    # Conservative estimate: 3 chars per token (to avoid underestimation)
    return total_chars // 3


def _get_context_limit(model: str) -> int:
    """Get context limit for a model."""
    if model is None:
        return DEFAULT_CONTEXT_LIMIT
    for model_prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if model_prefix in model.lower():
            return limit
    return DEFAULT_CONTEXT_LIMIT


def _truncate_messages_to_fit(
    system_messages: list,
    messages: list,
    tools: list,
    model: str,
) -> list:
    """Truncate messages to fit within context limit, keeping most recent.
    
    IMPORTANT: Preserves tool_call -> tool_response pairs to avoid API errors.
    """
    context_limit = _get_context_limit(model) - CONTEXT_BUFFER
    
    # Always keep system messages
    system_tokens = _estimate_tokens(system_messages, tools)
    available_tokens = context_limit - system_tokens
    
    logger.debug(f"Context check: model={model}, limit={context_limit}, system_tokens={system_tokens}, available={available_tokens}")
    
    if available_tokens <= 0:
        logger.warning(f"System messages alone exceed context limit! system_tokens={system_tokens}")
        return messages[-5:]  # Keep at least some recent messages
    
    # Check if truncation is needed
    current_tokens = _estimate_tokens(messages)
    logger.debug(f"Current message tokens: {current_tokens}, available: {available_tokens}")
    
    if current_tokens <= available_tokens:
        return messages  # No truncation needed
    
    logger.info(f"Truncation needed: {current_tokens} tokens > {available_tokens} available")
    
    # Group messages into "atomic units" that must stay together
    # A tool_call message must be followed by its tool_response(s)
    atomic_groups = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        group = [msg]
        
        # Check if this is an assistant message with tool_calls
        has_tool_calls = (
            hasattr(msg, 'tool_calls') and msg.tool_calls and 
            hasattr(msg, 'role') and getattr(msg, 'role', None) == 'assistant'
        )
        
        if has_tool_calls:
            # Collect all following tool messages that respond to this
            tool_call_ids = {tc.id for tc in msg.tool_calls if hasattr(tc, 'id')}
            j = i + 1
            while j < len(messages) and tool_call_ids:
                next_msg = messages[j]
                msg_role = getattr(next_msg, 'role', None)
                if msg_role == 'tool':
                    group.append(next_msg)
                    # Remove the tool_call_id if this message responds to it
                    msg_id = getattr(next_msg, 'tool_call_id', None) or getattr(next_msg, 'id', None)
                    tool_call_ids.discard(msg_id)
                    j += 1
                else:
                    break
            i = j
        else:
            i += 1
        
        atomic_groups.append(group)
    
    # Now truncate by atomic groups, keeping most recent
    truncated_groups = []
    running_tokens = 0
    
    for group in reversed(atomic_groups):
        group_tokens = _estimate_tokens(group)
        if running_tokens + group_tokens <= available_tokens:
            truncated_groups.insert(0, group)
            running_tokens += group_tokens
        else:
            logger.debug(f"Skipping message group with {group_tokens} tokens")
    
    # Flatten groups back to messages
    truncated = []
    for group in truncated_groups:
        truncated.extend(group)
    
    if len(truncated) < len(messages):
        removed = len(messages) - len(truncated)
        logger.warning(
            f"Context truncation: removed {removed} oldest messages "
            f"(estimated {current_tokens} -> {running_tokens} tokens) to fit {context_limit + CONTEXT_BUFFER} model limit"
        )
    
    # Safety check - if we still have too many, be more aggressive but keep pairs
    final_tokens = _estimate_tokens(truncated)
    if final_tokens > available_tokens and len(truncated_groups) > 2:
        logger.warning(f"Still over limit after truncation ({final_tokens} > {available_tokens}), removing more groups...")
        # Keep only the last half of groups
        truncated_groups = truncated_groups[len(truncated_groups)//2:]
        truncated = []
        for group in truncated_groups:
            truncated.extend(group)
        logger.warning(f"Aggressive truncation: kept last {len(truncated)} messages")
    
    return truncated

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class LLMAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class LLMAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        
        # Truncate messages if they exceed context limit
        truncated_messages = _truncate_messages_to_fit(
            system_messages=state.system_messages,
            messages=state.messages,
            tools=self.tools,
            model=self.llm,
        )
        
        messages = state.system_messages + truncated_messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed


AGENT_GT_INSTRUCTION = """
You are testing that our user simulator is working correctly.
User simulator will have an issue for you to solve.
You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
These steps involve either taking an action or asking the user to take an action.

In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_GT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<resolution_steps>
{resolution_steps}
</resolution_steps>
""".strip()


class LLMGTAgent(LocalAgent[LLMAgentState]):
    """
    An GroundTruth agent that can be used to solve a task.
    This agent will receive the expected actions.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
    ):
        """
        Initialize the LLMAgent.
        If provide_function_args is True, the resolution steps will include the function arguments.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.provide_function_args = provide_function_args

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Only the tasks that require at least one action are valid.
        """
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=AGENT_GT_INSTRUCTION,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        
        # Truncate messages if they exceed context limit
        truncated_messages = _truncate_messages_to_fit(
            system_messages=state.system_messages,
            messages=state.messages,
            tools=self.tools,
            model=self.llm,
        )
        
        messages = state.system_messages + truncated_messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed

    def make_agent_instructions_from_actions(self) -> str:
        """
        Make agent instructions from a list of actions
        """
        lines = []
        for i, action in enumerate(self.task.evaluation_criteria.actions):
            lines.append(
                f"[Step {i + 1}] {self.make_agent_instructions_from_action(action=action, include_function_args=self.provide_function_args)}"
            )
        return "\n".join(lines)

    @classmethod
    def make_agent_instructions_from_action(
        cls, action: Action, include_function_args: bool = False
    ) -> str:
        """
        Make agent instructions from an action.
        If the action is a user action, returns instructions for the agent to give to the user.
        If the action is an agent action, returns instructions for the agent to perform the action.
        """
        if action.requestor == "user":
            if include_function_args:
                return f"Instruct the user to perform the following action: {action.get_func_format()}."
            else:
                return f"User action: {action.name}."
        elif action.requestor == "assistant":
            if include_function_args:
                return f"Perform the following action: {action.get_func_format()}."
            else:
                return f"Assistant action: {action.name}."
        else:
            raise ValueError(f"Unknown action requestor: {action.requestor}")


AGENT_SOLO_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
You will be provided with a ticket that contains the user's request.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.
Stop when you consider that you have solved the ticket.
To do so, send a message containing a single tool call to the `{stop_function_name}` tool. Do not include any other tool calls in this last message.

Always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT_SOLO = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
""".strip()


class LLMSoloAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task without any interaction with the customer.
    The task need to specify a ticket format.
    """

    STOP_FUNCTION_NAME = "done"
    TRANSFER_TOOL_NAME = "transfer_to_human_agents"
    STOP_TOKEN = "###STOP###"

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.llm = llm
        self.llm_args = llm_args if llm_args is not None else {}
        self.add_stop_tool()
        self.validate_tools()

    def add_stop_tool(self) -> None:
        """Add the stop tool to the tools."""

        def done() -> str:
            """Call this function when you are done with the task."""
            return self.STOP_TOKEN

        self.tools.append(as_tool(done))

    def validate_tools(self) -> None:
        """Check if the tools are valid."""
        tool_names = {tool.name for tool in self.tools}
        if self.TRANSFER_TOOL_NAME not in tool_names:
            logger.warning(
                f"Tool {self.TRANSFER_TOOL_NAME} not found in tools. This tool is required for the agent to transfer the user to a human agent."
            )
        if self.STOP_FUNCTION_NAME not in tool_names:
            raise ValueError(f"Tool {self.STOP_FUNCTION_NAME} not found in tools.")

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Task should contain a ticket and evaluation criteria.
        If the task contains an initial state, the message history should only contain tool calls and responses.
        """
        if task.initial_state is not None:
            message_history = task.initial_state.message_history or []
            for message in message_history:
                if isinstance(message, UserMessage):
                    return False
                if isinstance(message, AssistantMessage) and not message.is_tool_call():
                    return False
            return True
        if task.ticket is None:
            return False
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        agent_instruction = AGENT_SOLO_INSTRUCTION.format(
            stop_function_name=self.STOP_FUNCTION_NAME,
            stop_token=self.STOP_TOKEN,
        )
        return SYSTEM_PROMPT_SOLO.format(
            agent_instruction=agent_instruction,
            domain_policy=self.domain_policy,
            ticket=self.task.ticket,
        )

    def _check_if_stop_toolcall(self, message: AssistantMessage) -> AssistantMessage:
        """Check if the message is a stop message.
        If the message contains a tool call with the name STOP_FUNCTION_NAME, then the message is a stop message.
        """
        is_stop = False
        for tool_call in message.tool_calls:
            if tool_call.name == self.STOP_FUNCTION_NAME:
                is_stop = True
                break
        if is_stop:
            message.content = self.STOP_TOKEN
            message.tool_calls = None
        return message

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message."""
        if message.content is None:
            return False
        return cls.STOP_TOKEN in message.content

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: Optional[ValidAgentInputMessage], state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, UserMessage):
            raise ValueError("LLMSoloAgent does not support user messages.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is None:
            assert len(state.messages) == 0, "Message history should be empty"
        else:
            state.messages.append(message)
        
        # Truncate messages if they exceed context limit
        truncated_messages = _truncate_messages_to_fit(
            system_messages=state.system_messages,
            messages=state.messages,
            tools=self.tools,
            model=self.llm,
        )
        
        messages = state.system_messages + truncated_messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            tool_choice="required",
            **self.llm_args,
        )
        if not assistant_message.is_tool_call():
            raise ValueError("LLMSoloAgent only supports tool calls.")
        message = self._check_if_stop_toolcall(assistant_message)
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed
