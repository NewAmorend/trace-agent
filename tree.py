"""Trace tree building and rendering."""

from models import NormalizedStep, TraceNode


def build_trace_tree(steps: list[NormalizedStep]) -> list[TraceNode]:
    """
    Build a tree from normalized steps.

    State-changing steps create new state nodes.
    Exploration steps are added to the current state.
    """
    if not steps:
        return []

    # Create root state (State 0)
    root = TraceNode(state_id=0)
    states = [root]
    current_state = root

    for step in steps:
        # Add step to current state
        current_state.steps.append(step)

        # If state-changing, create new state
        if step.state_change:
            new_state = TraceNode(state_id=len(states))
            current_state.children.append(new_state)
            states.append(new_state)
            current_state = new_state

    return states


def render_trace_tree(nodes: list[TraceNode]) -> str:
    """Render trace tree as Markdown."""
    if not nodes:
        return "# Trace Tree\n\nNo steps found.\n"

    lines = ["# Trace Tree\n"]

    for state in nodes:
        lines.append(f"\nState {state.state_id}")

        for step in state.steps:
            # Build label: [stage | action_type | explore/state_change]
            change_label = "state_change" if step.state_change else "explore"
            label = f"[{step.stage} | {step.action_type} | {change_label}]"

            lines.append(f"  - Step {step.step_id} {label} {step.action}")

        # Show transition to next state
        if state.children:
            next_state = state.children[0]
            lines.append(f"    -> State {next_state.state_id}")

    return "\n".join(lines)
